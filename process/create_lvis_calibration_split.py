#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


DEFAULT_SEED = 2024
DEFAULT_CALIB_SIZE = 512
REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create deterministic held-out LVIS calibration and training splits."
    )
    parser.add_argument(
        "--input",
        default="data/lvis_clear_75_60.json",
        help="path to the processed LVIS annotation file",
    )
    parser.add_argument(
        "--calib-output",
        default="data/lvis_clear_75_60_calib_512.json",
        help="path to write the held-out calibration split",
    )
    parser.add_argument(
        "--train-output",
        default="data/lvis_clear_75_60_train_excluding_calib_512.json",
        help="path to write the training split with calibration images removed",
    )
    parser.add_argument(
        "--calib-size",
        default=DEFAULT_CALIB_SIZE,
        type=int,
        help="number of images to assign to the held-out calibration split",
    )
    parser.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        type=int,
        help="random seed used after sorting images by id",
    )
    return parser.parse_args()


def load_dataset(path: Path):
    with path.open("r") as handle:
        return json.load(handle)


def select_calibration_ids(images, calib_size: int, seed: int):
    sorted_images = sorted(images, key=lambda image: image["id"])
    image_ids = [image["id"] for image in sorted_images]
    if calib_size <= 0 or calib_size >= len(image_ids):
        raise ValueError(f"Expected calib size in [1, {len(image_ids) - 1}], got {calib_size}")
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    return set(image_ids[:calib_size])


def build_subset(dataset, selected_image_ids):
    selected_image_ids = set(selected_image_ids)
    subset = {key: value for key, value in dataset.items() if key not in {"images", "annotations", "videos"}}

    subset["images"] = sorted(
        [image for image in dataset["images"] if image["id"] in selected_image_ids],
        key=lambda image: image["id"],
    )
    subset["annotations"] = sorted(
        [annotation for annotation in dataset["annotations"] if annotation["image_id"] in selected_image_ids],
        key=lambda annotation: annotation["id"],
    )
    selected_video_ids = {image["video_id"] for image in subset["images"]}
    subset["videos"] = sorted(
        [video for video in dataset["videos"] if video["id"] in selected_video_ids],
        key=lambda video: video["id"],
    )
    return subset


def validate_split(original_dataset, calib_dataset, train_dataset):
    original_image_ids = {image["id"] for image in original_dataset["images"]}
    calib_image_ids = {image["id"] for image in calib_dataset["images"]}
    train_image_ids = {image["id"] for image in train_dataset["images"]}

    overlap = calib_image_ids & train_image_ids
    if overlap:
        raise ValueError(f"Calibration/train overlap detected for {len(overlap)} images")
    if calib_image_ids | train_image_ids != original_image_ids:
        raise ValueError("Calibration/train image ids do not cover the original dataset exactly")

    for name, subset in (("calib", calib_dataset), ("train", train_dataset)):
        subset_image_ids = {image["id"] for image in subset["images"]}
        subset_video_ids = {video["id"] for video in subset["videos"]}
        for annotation in subset["annotations"]:
            if annotation["image_id"] not in subset_image_ids:
                raise ValueError(f"{name} annotation {annotation['id']} references a missing image")
            if annotation["video_id"] not in subset_video_ids:
                raise ValueError(f"{name} annotation {annotation['id']} references a missing video")
        for image in subset["images"]:
            if image["video_id"] not in subset_video_ids:
                raise ValueError(f"{name} image {image['id']} references a missing video")


def write_dataset(path: Path, dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(dataset, handle)


def main():
    args = parse_args()
    input_path = (REPO_ROOT / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    calib_output_path = (
        (REPO_ROOT / args.calib_output).resolve()
        if not Path(args.calib_output).is_absolute()
        else Path(args.calib_output)
    )
    train_output_path = (
        (REPO_ROOT / args.train_output).resolve()
        if not Path(args.train_output).is_absolute()
        else Path(args.train_output)
    )

    dataset = load_dataset(input_path)
    calib_image_ids = select_calibration_ids(dataset["images"], args.calib_size, args.seed)
    all_image_ids = {image["id"] for image in dataset["images"]}
    train_image_ids = all_image_ids - calib_image_ids

    calib_dataset = build_subset(dataset, calib_image_ids)
    train_dataset = build_subset(dataset, train_image_ids)
    validate_split(dataset, calib_dataset, train_dataset)

    write_dataset(calib_output_path, calib_dataset)
    write_dataset(train_output_path, train_dataset)

    overlap = {image["id"] for image in calib_dataset["images"]} & {image["id"] for image in train_dataset["images"]}
    print(f"source_images={len(dataset['images'])}")
    print(f"calib_images={len(calib_dataset['images'])}")
    print(f"train_images={len(train_dataset['images'])}")
    print(f"overlap_images={len(overlap)}")
    print(f"calib_output={calib_output_path}")
    print(f"train_output={train_output_path}")


if __name__ == "__main__":
    main()
