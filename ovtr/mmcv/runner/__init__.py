import torch.distributed as dist


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


class EvalHook:
    def __init__(self, dataloader=None, interval=1, save_best=False, **kwargs):
        self.dataloader = dataloader
        self.interval = interval
        self.save_best = save_best

    def _should_evaluate(self, runner):
        return True

    def evaluate(self, runner, results):
        return None

    def _save_ckpt(self, runner, key_score):
        return None


class DistEvalHook(EvalHook):
    def __init__(
        self,
        dataloader=None,
        interval=1,
        tmpdir=None,
        gpu_collect=False,
        broadcast_bn_buffer=True,
        save_best=False,
        **kwargs,
    ):
        super().__init__(dataloader=dataloader, interval=interval, save_best=save_best, **kwargs)
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect
        self.broadcast_bn_buffer = broadcast_bn_buffer
