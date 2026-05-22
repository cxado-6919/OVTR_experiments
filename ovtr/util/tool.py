import torch
import numpy as np
from .utils import clean_state_dict
from models.quant_utils import apply_ovtr_quant_state_dict, materialize_checkpoint_bias_parameters


MCIP_CHECKPOINT_KEY_MARKERS = (
    "track_embed.gate_mlp.",
    "track_embed.memory_img_proj.",
    "track_embed.memory_sem_proj.",
    "track_embed.motion_scale",
)

OV_DPTD_CHECKPOINT_KEY_MARKERS = (
    "ov_dptd_",
    "dptd_visual_memory_proj.",
)


def is_mcip_checkpoint_key(name):
    if name.startswith("module."):
        name = name[7:]
    return any(marker in name for marker in MCIP_CHECKPOINT_KEY_MARKERS)


def is_ov_dptd_checkpoint_key(name):
    if name.startswith("module."):
        name = name[7:]
    return any(marker in name for marker in OV_DPTD_CHECKPOINT_KEY_MARKERS)




def maybe_warn_or_reinit_dead_ov_dptd_semantic_gate(
    model,
    reinit=False,
    init_std=1e-3,
):
    module = model.module if hasattr(model, "module") else model
    decoder = getattr(getattr(module, "transformer", None), "decoder", None)
    if decoder is None:
        return []
    if not getattr(decoder, "use_ov_dptd", False) or getattr(decoder, "ov_dptd_fusion", None) != "semantic_gate":
        return []
    alpha = getattr(decoder, "ov_dptd_gate_alpha", None)
    if alpha is None or float(alpha.detach().abs().max().item()) > 1e-12:
        return []
    if reinit and float(init_std) <= 0.0:
        raise RuntimeError("ov_dptd_semantic_gate_id_proj_init_std must be > 0 for dead semantic_gate reinit.")

    dead_layers = []
    for layer_id, proj in enumerate(getattr(decoder, "ov_dptd_ofa_id_proj", [])):
        weight_norm = float(proj.weight.detach().float().norm().item())
        if weight_norm <= 1e-12:
            dead_layers.append(layer_id)
            print(
                "Warning: OV-DPTD semantic_gate dead id projection detected "
                f"at decoder layer {layer_id}; alpha=0 and ov_dptd_ofa_id_proj.weight norm=0."
            )
            if reinit:
                torch.nn.init.normal_(proj.weight, mean=0.0, std=float(init_std))
                torch.nn.init.zeros_(proj.bias)
                print(
                    "Reinitialized OV-DPTD semantic_gate id projection "
                    f"at decoder layer {layer_id} with std={float(init_std)}."
                )
    return dead_layers

def load_model(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None, allow_mcip_missing=False, ov_dptd_reinit_dead_semantic_gate_id_proj=False, ov_dptd_semantic_gate_id_proj_init_std=1e-3):
    start_epoch = 0
    checkpoint = torch.load(
        model_path,
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    print(f'loaded {model_path}')
    
    state_dict = clean_state_dict(checkpoint['model'])
    state_dict = apply_ovtr_quant_state_dict(model, state_dict)
    materialized = materialize_checkpoint_bias_parameters(model, state_dict)
    if materialized:
        print(f"Materialized {materialized} checkpoint bias parameters.")
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            if "_group_a_" not in k:
                print('Drop parameter {}.'.format(k))
    allowed_mcip_missing = []
    allowed_ov_dptd_missing = []
    for k in model_state_dict:
        if not (k in state_dict):
            if "_ovtr_quant_" in k:
                # Quant tensors are loaded directly into the patched modules by
                # apply_ovtr_quant_state_dict(); leaving them as missing here
                # avoids feeding empty default observer buffers back through
                # torch.load_state_dict().
                continue
            if allow_mcip_missing and is_mcip_checkpoint_key(k):
                allowed_mcip_missing.append(k)
            elif is_ov_dptd_checkpoint_key(k):
                allowed_ov_dptd_missing.append(k)
            else:
                print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    if allowed_mcip_missing:
        print('Allowed missing M-CIP Keys: {}'.format(allowed_mcip_missing))
    if allowed_ov_dptd_missing:
        print('Allowed missing OV-DPTD Keys: {}'.format(allowed_ov_dptd_missing))
    maybe_warn_or_reinit_dead_ov_dptd_semantic_gate(
        model,
        reinit=ov_dptd_reinit_dead_semantic_gate_id_proj,
        init_std=ov_dptd_semantic_gate_id_proj_init_std,
    )
    print("|| Weights have been checked completely ||")
    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model



