# from: https://github.com/csrhddlam/pytorch-checkpoint

import torch
import warnings


def detach_variable(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


def check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        for i in range(len(ctx.input_tensors)):
            temp = ctx.input_tensors[i]
            ctx.input_tensors[i] = temp.detach()
            ctx.input_tensors[i].requires_grad = temp.requires_grad
        with torch.enable_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        to_autograd = []
        for i in range(len(ctx.input_tensors)):
            if ctx.input_tensors[i].requires_grad:
                to_autograd.append(ctx.input_tensors[i])

        grad_pairs = [
            (output_tensor, output_grad)
            for output_tensor, output_grad in zip(output_tensors, output_grads)
            if isinstance(output_tensor, torch.Tensor)
            and output_tensor.requires_grad
            and output_grad is not None
        ]
        if not grad_pairs:
            total_inputs = len(ctx.input_tensors) + len(ctx.input_params)
            return (None, None) + (None,) * total_inputs

        output_tensors, output_grads = zip(*grad_pairs)
        input_grads = torch.autograd.grad(output_tensors, to_autograd + ctx.input_params, output_grads, allow_unused=True)
        input_grads = list(input_grads)
        for i in range(len(ctx.input_tensors)):
            if not ctx.input_tensors[i].requires_grad:
                input_grads.insert(i, None)
        return (None, None) + tuple(input_grads)
