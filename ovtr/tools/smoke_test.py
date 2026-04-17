import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mmcv  # noqa: F401
import mmdet  # noqa: F401

from util.slconfig import SLConfig
from models.ms_deform_attn import HAS_MSDA_EXT, MultiScaleDeformableAttention


def run_attention_smoke(device):
    module = MultiScaleDeformableAttention(
        embed_dim=32,
        num_heads=4,
        num_levels=2,
        num_points=2,
        batch_first=True,
    ).to(device)

    query = torch.randn(1, 3, 32, device=device)
    value = torch.randn(1, 5, 32, device=device)
    spatial_shapes = torch.tensor([[2, 2], [1, 1]], dtype=torch.long, device=device)
    level_start_index = torch.tensor([0, 4], dtype=torch.long, device=device)
    reference_points = torch.rand(1, 3, 2, 2, device=device)

    out = module(
        query=query,
        value=value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )
    assert out.shape == (1, 3, 32)


def main():
    cfg = SLConfig.fromfile("./config/ovtr_lite_test.py")
    assert cfg.data.test.type == "TaoDataset"

    cpu = torch.device("cpu")
    run_attention_smoke(cpu)

    print(f"HAS_MSDA_EXT={HAS_MSDA_EXT}")
    print(f"CPU smoke passed for {Path(cfg.filename).name}")

    if torch.cuda.is_available():
        run_attention_smoke(torch.device("cuda"))
        print("CUDA smoke passed")
    else:
        print("CUDA unavailable, skipped CUDA smoke")


if __name__ == "__main__":
    main()
