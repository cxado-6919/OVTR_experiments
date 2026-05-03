import torch
import numpy as np
from .utils import clean_state_dict
from models.quant_utils import apply_ovtr_quant_state_dict, materialize_checkpoint_bias_parameters


def load_model(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None):
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
    for k in model_state_dict:
        if not (k in state_dict):
            if "_ovtr_quant_" in k:
                # Quant tensors are loaded directly into the patched modules by
                # apply_ovtr_quant_state_dict(); leaving them as missing here
                # avoids feeding empty default observer buffers back through
                # torch.load_state_dict().
                continue
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
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



