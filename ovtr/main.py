# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from util.events import EventStorage, TensorboardXWriter
from util.tool import load_model
from util.quantization import (
    add_quant_args,
    build_quant_manifest,
    build_quant_calibration_loader,
    calibrate_quant_controller_on_val_loader,
    enable_loaded_quantization,
    prepare_quant_model_for_calibration,
    setup_quant_controller,
    write_quant_manifest,
)
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from engine import train_one_epoch_mot
from models import build_model
from models.quant_utils import (
    apply_ovtr_quant_state_dict,
    is_partition_trainable_param,
    is_quant_trainable_param,
    materialize_checkpoint_bias_parameters,
)

from util.slconfig import SLConfig


def should_use_manual_grad_sync(args):
    if not args.distributed or not str(args.device).startswith("cuda"):
        return False

    force_ddp = os.environ.get("OVTR_FORCE_DDP", "0") == "1"
    if force_ddp:
        return False

    force_manual = os.environ.get("OVTR_MANUAL_GRAD_SYNC", "0") == "1"
    if force_manual:
        return True

    # Blackwell DDP can hit CUDA illegal memory access during AccumulateGrad/NCCL.
    major, _ = torch.cuda.get_device_capability(args.gpu)
    return major >= 12


def get_args_parser():
    parser = argparse.ArgumentParser('OVTR Tracker', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets',], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument("--lr_drop", type=int, nargs='*')
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--clip_gradients', action='store_true')
    parser.add_argument('--clip_gradients_type', default='full_model', type=str)
    
    parser.add_argument("--save_period", default=1, type=int)
    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--accurate_ratio', default=False, action='store_true')


    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--num_anchors', default=1, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--mix_match', action='store_true',)
    parser.add_argument('--atss_topk', default=9, type=int)
    parser.add_argument('--minus_std', action='store_true',)
    parser.add_argument('--set_cost_class', default=3, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument("--align_loss_coef", default=2, type=float)
    parser.add_argument("--align_pre_loss_coef", default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='lvis')

    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # parser.add_argument('--eval', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', 
                        help='whether to cache images on memory')

    # end-to-end mot settings.
    parser.add_argument('--save_path', default='results.json')

    parser.add_argument('--max_size', default=1333, type=int)
    parser.add_argument('--val_width', default=800, type=int)
    parser.add_argument('--filter_ignore', action='store_true')

    parser.add_argument('--track_query_iteration', default='CIP', type=str,
                        help="")
    parser.add_argument('--sample_mode', type=str, default='fixed_interval')
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--random_drop', type=float, default=0)
    parser.add_argument('--fp_ratio', type=float, default=0)
    parser.add_argument('--merger_dropout', type=float, default=0.1)
    parser.add_argument('--update_query_pos', action='store_true')
    parser.add_argument('--max_objs', type=int, default=10)
    parser.add_argument('--filter_low_quality', default=False, action='store_true')
    parser.add_argument('--shift_invalid_sample', default=False, action='store_true')
    parser.add_argument('--low_quality_threshold', default=0.5, type=float)
    parser.add_argument('--high_resolution_training', default=False, action='store_true')
    parser.add_argument('--n_keep', default=256, type=int,
                        help="Number of coeffs to be remained")
    parser.add_argument('--gt_mask_len', default=128, type=int,
                        help="Size of target mask")
    parser.add_argument('--checkpoint', default=False, action='store_true')

    parser.add_argument('--sampler_steps', type=int, nargs='*')
    parser.add_argument('--sampler_lengths', type=int, nargs='*')
    parser.add_argument('--exp_name', default='submit', type=str)
    parser.add_argument('--occlusion_class', action='store_true',
                        help="whether regard occluded track as an unique class in track classification.")

    parser.add_argument("--config_file", default="./config/ovtr_5_frame_train.py", type=str)
    parser.add_argument('--calculate_negative_samples', default=False, action='store_true')
    parser.add_argument(
        '--pretrained', '--pretrain',
        dest='pretrained',
        default=None,
        help='path to pretrained weights to load before training/evaluation',
    )
    parser.add_argument('--max_len', default=100, type=int)
    parser.add_argument('--lvis_anno', default="lvis_v1_train.json", type=str)

    # evaluation
    parser.add_argument('--score_thresh', type=float, nargs='*')
    parser.add_argument('--filter_score_thresh', type=float, nargs='*')
    parser.add_argument('--ious_thresh', type=float, nargs='*')
    parser.add_argument('--prob_threshold', default=0.6, type=float)
    parser.add_argument('--area_threshold', default=100, type=int)
    parser.add_argument('--miss_tolerance', type=int, nargs='*')
    parser.add_argument('--maximum_quantity', default=160, type=int)
    parser.add_argument('--key_word', default=None, type=str)
    parser.add_argument('--vis_output', default=None, type=str)
    parser.add_argument('--vis_points', default=None, type=str)
    parser.add_argument('--eval', default=['track'], type=str, nargs='+')
    parser.add_argument('--eval_options', type=json.loads, default='{"resfile_path": "results/ovtrack_teta_results/"}')
    parser.add_argument('--result_path_track', default=None, type=str)
    add_quant_args(parser)
    return parser


def main(args):
    t_s = time.time()
    utils.init_distributed_mode(args)
    if getattr(args, "quant_deploy", "none") == "int_msda":
        raise RuntimeError("--quant_deploy int_msda is eval/inference only; do not use it with main.py training.")
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    if args.distributed and str(args.device).startswith("cuda"):
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device(args.device)

    def ddp_debug(msg):
        print(
            f"[DDP-DBG][rank={utils.get_rank()}][local_rank={getattr(args, 'gpu', 0)}] {msg}",
            flush=True,
        )

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cfg = SLConfig.fromfile(args.config_file)
    cfg.device = str(device)
    if args.quant_mode == "qat" and args.resume is None and args.epochs == 50:
        args.epochs = 1
        print("[Quant] Defaulting QAT fine-tuning to 1 epoch", flush=True)
    if args.quant_mode == "qat":
        args.sampler_steps = []
        args.sampler_lengths = [5]
        allow_batched_qat = args.quant_qat_allow_batch and args.batch_size > 1
        if allow_batched_qat:
            cfg.use_transformer_ckpt = False
            cfg.use_checkpoint_track = False
            print(
                "[Quant] Experimental QAT batch mode enabled for batch_size > 1; checkpoint forcing is disabled.",
                flush=True,
            )
        else:
            if args.quant_qat_allow_batch and args.batch_size <= 1:
                print(
                    "[Quant] Ignoring quant_qat_allow_batch because batch_size <= 1; keeping QAT checkpointing enabled.",
                    flush=True,
                )
            if not getattr(cfg, "use_transformer_ckpt", False):
                cfg.use_transformer_ckpt = True
            if not getattr(cfg, "use_checkpoint_track", False):
                cfg.use_checkpoint_track = True
            print(
                "[Quant] Enabling transformer checkpointing and frame-wise checkpointing for QAT",
                flush=True,
            )
        print("[Quant] Forcing QAT training to fixed 5-frame sampling", flush=True)

    ddp_debug("before build_model")
    model, criterion = build_model(args, cfg)
    ddp_debug("after build_model")
    model.to(device)
    ddp_debug(f"after model.to({device})")

    model_without_ddp = model
    quant_controller = setup_quant_controller(model_without_ddp, args)

    dataset_train = build_dataset(image_set='train', args=args, cfg=cfg.data.train)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    
    datasets2collate_fn = {
        'lvis_generated_img_seqs': utils.mot_collate_fn
    }
    collate_fn = datasets2collate_fn[args.dataset_file]
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out
    
    output_dir = Path(args.output_dir)

    def build_default_param_dicts():
        return [
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

    def build_qat_param_dicts():
        return [
            {
                "params": [
                    p
                    for n, p in model_without_ddp.named_parameters()
                    if is_quant_trainable_param(n) and p.requires_grad
                ],
                "lr": args.lr * 0.1,
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in model_without_ddp.named_parameters()
                    if not is_quant_trainable_param(n)
                    and not match_name_keywords(n, args.lr_backbone_names)
                    and not match_name_keywords(n, args.lr_linear_proj_names)
                    and p.requires_grad
                ],
                "lr": args.lr,
            },
            {
                "params": [
                    p
                    for n, p in model_without_ddp.named_parameters()
                    if not is_quant_trainable_param(n)
                    and match_name_keywords(n, args.lr_backbone_names)
                    and p.requires_grad
                ],
                "lr": args.lr_backbone,
            },
            {
                "params": [
                    p
                    for n, p in model_without_ddp.named_parameters()
                    if not is_quant_trainable_param(n)
                    and match_name_keywords(n, args.lr_linear_proj_names)
                    and p.requires_grad
                ],
                "lr": args.lr * args.lr_linear_proj_mult,
            },
        ]

    freeze_ori = []
    if args.quant_mode == "qat":
        for _, para in model.named_parameters():
            para.requires_grad_(False)
        for name, para in model.named_parameters():
            if (
                is_partition_trainable_param(name, args.quant_partition)
                or is_quant_trainable_param(name)
            ):
                para.requires_grad_(True)
        param_dicts = [group for group in build_qat_param_dicts() if len(group["params"]) > 0]
    else:
        param_dicts = build_default_param_dicts()

        if (cfg.train_tracking_keep is not None) and (cfg.initial_grad):
            for name, para in model.named_parameters():
                for keyw in cfg.train_tracking_keep:
                    if keyw in name:
                        para.requires_grad_(False)
                        break

        if (cfg.train_tracking_only is not None) and (cfg.initial_grad) and (args.resume is None):
            for _, para in model.named_parameters():
                para.requires_grad_(False)
            for name, para in model.named_parameters():
                for keyw in cfg.train_tracking_only:
                    if keyw in name:
                        para.requires_grad_(True)
                        break

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print("ori_requires_grad: False ", name)
            freeze_ori.append(name)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("requires_grad: True ", name)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print("requires_grad: False ", name)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = None
    lr_scheduler = None
    if args.quant_mode != "ptq":
        if args.sgd:
            optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                          weight_decay=args.weight_decay)

        if len(args.lr_drop)==1:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop[0])
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    if args.frozen_weights is not None:
        ddp_debug(f"before frozen_weights load: {args.frozen_weights}")
        checkpoint = torch.load(args.frozen_weights, map_location="cpu", weights_only=False)
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
        ddp_debug("after frozen_weights load")

    if args.pretrained is not None:
        ddp_debug(f"before pretrained load: {args.pretrained}")
        load_model(model_without_ddp, args.pretrained)
        ddp_debug("after pretrained load")

    if args.resume:
        ddp_debug(f"before resume load: {args.resume}")
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_state = apply_ovtr_quant_state_dict(model_without_ddp, checkpoint['model'])
        materialized = materialize_checkpoint_bias_parameters(model_without_ddp, model_state)
        if materialized:
            print(f"Materialized {materialized} checkpoint bias parameters.")
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(model_state, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if optimizer is not None and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            if len(args.lr_drop)==1:
                args.override_resumed_lr_drop = True
            else:
                args.override_resumed_lr_drop = False
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                if len(args.lr_drop)==1:
                    lr_scheduler.step_size = args.lr_drop[0]
                else:
                    lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        ddp_debug("after resume load")

    def save_calibrated_checkpoint(filename: str) -> None:
        if not args.output_dir:
            return
        checkpoint_path = output_dir / filename
        payload = {
            'model': model_without_ddp.state_dict(),
            'args': args,
            'quant_meta': build_quant_manifest(model_without_ddp, args),
        }
        if optimizer is not None:
            payload['optimizer'] = optimizer.state_dict()
        if lr_scheduler is not None:
            payload['lr_scheduler'] = lr_scheduler.state_dict()
        utils.save_on_master(payload, checkpoint_path)

    quant_state_loaded = False
    if quant_controller is not None:
        quant_state_loaded = enable_loaded_quantization(model_without_ddp, require_state=False)
        if not quant_state_loaded:
            prepare_quant_model_for_calibration(
                model_without_ddp,
                args,
                quant_state_loaded=quant_state_loaded,
            )

    if args.quant_mode == "ptq":
        if quant_controller is None:
            raise ValueError("PTQ requested but quant controller was not initialized.")
        if not quant_state_loaded:
            data_loader_calib = build_quant_calibration_loader(args, cfg)
            calibrated = calibrate_quant_controller_on_val_loader(
                model_without_ddp,
                data_loader_calib,
                device,
                args.quant_calib_samples,
                args=args,
            )
            quant_controller.enable_quantization()
            print(f"[Quant] PTQ calibration complete on {calibrated} samples", flush=True)
        save_calibrated_checkpoint("checkpoint_quant_calibrated.pth")
        write_quant_manifest(model_without_ddp, args)
        print("[Quant] PTQ flow does not start training; exiting after calibration/runtime setup.", flush=True)
        return

    if args.quant_mode == "qat" and quant_controller is not None and not quant_state_loaded:
        data_loader_calib = build_quant_calibration_loader(args, cfg)
        calibrated = calibrate_quant_controller_on_val_loader(
            model_without_ddp,
            data_loader_calib,
            device,
            args.quant_calib_samples,
            args=args,
        )
        quant_controller.enable_qat()
        print(f"[Quant] QAT initialization complete on {calibrated} samples", flush=True)
        if args.quant_calibration_only:
            save_calibrated_checkpoint("checkpoint_quant_initialized.pth")
            write_quant_manifest(model_without_ddp, args)
            print("[Quant] Exiting after QAT initialization as requested.", flush=True)
            return

    if quant_controller is not None:
        write_quant_manifest(model_without_ddp, args)

    if args.distributed:
        args.manual_grad_sync = should_use_manual_grad_sync(args)
        if args.manual_grad_sync:
            ddp_debug("skipping DDP wrap; using manual gradient all-reduce fallback")
        else:
            ddp_debug("before DDP wrap")
            ddp_bucket_cap_mb = int(os.environ.get("OVTR_DDP_BUCKET_CAP_MB", "4"))
            model = torch.nn.parallel.DistributedDataParallel(
                model_without_ddp,
                device_ids=[args.gpu],
                output_device=args.gpu,
                find_unused_parameters=False,
                broadcast_buffers=False,
                gradient_as_bucket_view=False,
                bucket_cap_mb=ddp_bucket_cap_mb,
            )
            model_without_ddp = model.module
            ddp_debug(f"after DDP wrap (bucket_cap_mb={ddp_bucket_cap_mb})")
    else:
        args.manual_grad_sync = False

    t_e = time.time()
    print("Training started, preparation took {:.2f} seconds.".format(t_e - t_s))
    start_time = time.time()
    train_func = train_one_epoch_mot
    dataset_train.set_epoch(args.start_epoch)
    with EventStorage(args.start_epoch * len(dataset_train)) as storage:
        writer = None
        if args.vis and utils.is_main_process():
            writer = TensorboardXWriter(output_dir)
        for epoch in range(args.start_epoch, args.epochs):
            if (
                args.quant_mode != "qat"
                and (epoch == cfg.global_grad_allowed_epoch_track)
                and (cfg.initial_grad)
                and (args.resume is None)
            ):
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
            train_stats = train_func(
                model,
                criterion,
                data_loader_train,
                optimizer,
                device,
                epoch,
                args.clip_max_norm,
                writer=writer,
                clip_gradients=args.clip_gradients,
                manual_grad_sync=args.manual_grad_sync,
            )
            if lr_scheduler is not None and (args.quant_mode != "qat" or args.quant_use_scheduler):
                lr_scheduler.step()
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                if (epoch + 1) % args.save_period == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                        'quant_meta': build_quant_manifest(model_without_ddp, args),
                    }, checkpoint_path)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}

                if args.output_dir and utils.is_main_process():
                    with (output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                        
            dataset_train.step_epoch()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('OVTR training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)  

