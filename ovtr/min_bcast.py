import os
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl", init_method="env://")

    x = torch.tensor([rank + 1.0], device=device)
    print(f"[rank {rank}] before all_reduce: {x.item()}", flush=True)
    dist.all_reduce(x)
    torch.cuda.synchronize(device)
    print(f"[rank {rank}] after all_reduce: {x.item()}", flush=True)

    for n in [1, 1024, 196608]:
        t = torch.empty(n, device=device, dtype=torch.float32).normal_()
        print(f"[rank {rank}] before broadcast n={n}", flush=True)
        dist.broadcast(t, src=0)
        torch.cuda.synchronize(device)
        print(f"[rank {rank}] after broadcast n={n}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()