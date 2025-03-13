# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import datasets.samplers as samplers
import util.misc as utils
from datasets import build_dataset
from models import build_model

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from util.tool import load_model
from util.slconfig import SLConfig
import datetime


def get_args_parser():
    parser = argparse.ArgumentParser("OVTR detection pre-training", add_help=False)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--lr_backbone_names", default=["backbone.0"], type=str, nargs="+")
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    parser.add_argument(
        "--lr_linear_proj_names",
        default=["reference_points", "sampling_offsets"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--lr_drop", default=40, type=int)
    parser.add_argument("--lr_drop_epochs", default=None, type=int, nargs="+")
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )
    parser.add_argument("--save_period", default=1, type=int)
    parser.add_argument("--sgd", action="store_true")

    # Variants of Deformable DETR
    parser.add_argument("--with_box_refine", default=True, action="store_true")
    parser.add_argument("--two_stage", default=True, action="store_true")

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Segmentation
    parser.add_argument(
        "--masks", action="store_true", help="Train segmentation head if the flag is provided"
    )

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=3, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--cls_loss_coef", default=2, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--align_loss_coef", default=2, type=float)
    parser.add_argument("--align_pre_loss_coef", default=2, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)

    parser.add_argument("--prob", default=1, type=float)

    # dataset parameters
    parser.add_argument("--dataset_file", default="lvis")

    parser.add_argument("--output_dir", default="./output",
                        help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda",
                        help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default=None,
                        help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N",
                        help="start epoch")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--eval_period", default=1, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', 
                        help='whether to cache images on memory')
    parser.add_argument("--amp", default=False, action="store_true")

    parser.add_argument('--random_drop', type=float, default=0)

    parser.add_argument("--config_file", default="./config/ovtr_det_bs2_pretrain.py", type=str)
    parser.add_argument('--calculate_negative_samples', default=False, action='store_true')
    parser.add_argument('--pretrained', default=None, help='resume from checkpoint')
    parser.add_argument('--max_len', default=13, type=int)
    parser.add_argument('--lvis_anno', default="lvis_v1_train.json", type=str)
    return parser


def main(args):
    t_s = time.time()
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cfg = SLConfig.fromfile(args.config_file)
    cfg.device = "cuda" #if not cpu_only else "cpu"

    model, criterion = build_model(args, cfg)
    model.to(device)
    
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    dataset_train = build_dataset(image_set="train", args=args, cfg=cfg)
    
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and p.requires_grad
            ],
            "lr": args.lr,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
    ]


    freeze_ori = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print("ori_requires_grad: False ", name)
            freeze_ori.append(name)

    # Pre adjustable weights
    if (cfg.initial_grad_allowed is not None) and (cfg.initial_grad) and (args.resume is None):
        for name, para in model.named_parameters():
            para.requires_grad_(False)
        for name, para in model.named_parameters():
            for keyw in cfg.initial_grad_allowed:
                if keyw in name:
                    para.requires_grad_(True)
                else:
                    pass

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("requires_grad: True ", name)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print("requires_grad: False ", name)

    if args.sgd:
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    if args.pretrained is not None:
        model_without_ddp = load_model(model_without_ddp, args.pretrained)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint["model"], strict=False
        )
        unexpected_keys = [
            k
            for k in unexpected_keys
            if not (k.endswith("total_params") or k.endswith("total_ops"))
        ]
        if len(missing_keys) > 0:
            print("Missing Keys: {}".format(missing_keys))
        if len(unexpected_keys) > 0:
            print("Unexpected Keys: {}".format(unexpected_keys))
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            import copy

            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint["optimizer"])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg["lr"] = pg_old["lr"]
                pg["initial_lr"] = pg_old["initial_lr"]
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(
                    map(lambda group: group["initial_lr"], optimizer.param_groups)
                )
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint["epoch"] + 1

    t_e = time.time()
    print("Detection pretraining started, preparation took {:.2f} seconds.".format(t_e - t_s))
    start_time = time.time()
    begin_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    for epoch in range(args.start_epoch, args.epochs):
        if (epoch == cfg.global_grad_allowed_epoch) and (cfg.initial_grad) and (args.resume is None):
            for name, para in model.named_parameters():
                if args.distributed:
                        if name[7:] in freeze_ori:
                            continue
                        else:
                            para.requires_grad_(True)
                else:
                    if name in freeze_ori:
                        continue
                    else:
                        para.requires_grad_(True)
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print(f"requires_grad in epoch{epoch}: False ", name)
                    
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            args.masks,
            args.amp,
        )
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_period == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / f"log_{begin_time}.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("OVTR detection pre-training script", parents=[get_args_parser()])
    args = parser.parse_args()
    from engine_ov import train_one_epoch
    args.label_map=True

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)