#!/usr/bin/env python3
"""Generate OVTrack-style class-score bars for saved OVTR embedding examples."""

import argparse
import sys
from pathlib import Path
from typing import Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
OVTR_ROOT = REPO_ROOT / "ovtr"
if str(OVTR_ROOT) not in sys.path:
    sys.path.insert(0, str(OVTR_ROOT))

from util.list_LVIS import CLASSES
from util.quant_drift_analysis import write_embedding_class_score_artifacts_from_npz


DEFAULT_INPUT_DIR = "ovtr/results/embedding_viz_lite_qat_exp_a1_to_b_val/embedding_viz"
DEFAULT_CLASS_ANCHOR_PATH = "model_zoo/clip_image_embedding_all.pt"


def _resolve_path(value: str, *, must_exist: bool) -> Path:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, REPO_ROOT / path, OVTR_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    resolved = candidates[0].resolve()
    if must_exist:
        tried = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Path does not exist: {value}. Tried: {tried}")
    return resolved


def _iter_example_npzs(input_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in input_dir.glob("*/examples/*.npz")
        if not path.name.endswith("_class_scores.npz")
    )


def _load_anchor_tensor(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload.float()
    if isinstance(payload, dict):
        preferred_keys = (
            "image_embeddings",
            "class_embeddings",
            "text_features",
            "clip_image_embeddings",
            "embeddings",
        )
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, torch.Tensor) and value.ndim == 2:
                return value.float()
        for value in payload.values():
            if isinstance(value, torch.Tensor) and value.ndim == 2:
                return value.float()
    raise ValueError(f"Could not find a 2D class-anchor tensor in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Embedding visualization directory containing <class>/examples/*.npz files.",
    )
    parser.add_argument(
        "--class-anchor-path",
        default=DEFAULT_CLASS_ANCHOR_PATH,
        help="Path to the OVTR class-anchor tensor. Defaults to model_zoo/clip_image_embedding_all.pt.",
    )
    parser.add_argument(
        "--top-k-per-model",
        default=10,
        type=int,
        help="Top foreground classes per model included with the target class.",
    )
    parser.add_argument(
        "--max-foreground",
        default=12,
        type=int,
        help="Maximum foreground bars including the target class.",
    )
    parser.add_argument(
        "--temperature",
        default=0.007,
        type=float,
        help="Softmax temperature used for target probability diagnostics.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate PNG/CSV files even when they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = _resolve_path(args.input_dir, must_exist=True)
    anchor_path = _resolve_path(args.class_anchor_path, must_exist=True)
    class_anchors = _load_anchor_tensor(anchor_path)

    npz_paths = list(_iter_example_npzs(input_dir))
    written = 0
    skipped = 0
    failed = []
    for npz_path in npz_paths:
        try:
            did_write = write_embedding_class_score_artifacts_from_npz(
                npz_path,
                class_anchors,
                class_names=CLASSES,
                score_topk=args.top_k_per_model,
                score_max_bars=args.max_foreground,
                score_temperature=args.temperature,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # noqa: BLE001 - report all failed examples and continue.
            failed.append((npz_path, exc))
            continue
        if did_write:
            written += 1
        else:
            skipped += 1

    print(f"input_dir={input_dir}")
    print(f"class_anchor_path={anchor_path}")
    print(f"class_anchor_shape={tuple(class_anchors.shape)} foreground_class_count={len(CLASSES)}")
    print(f"npz_examples={len(npz_paths)} written={written} skipped={skipped} failed={len(failed)}")
    if class_anchors.shape[0] <= len(CLASSES):
        print("background=omitted (no extra background anchor row)")
    else:
        print("background=shown (class anchors include extra rows beyond named foreground classes)")
    for npz_path, exc in failed[:10]:
        print(f"FAILED {npz_path}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
