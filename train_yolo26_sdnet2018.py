from __future__ import annotations

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


@dataclass(frozen=True)
class ImageSample:
    image_path: Path
    surface: str
    label_dir: str
    cracked: bool


LABEL_DIRS = {
    "D": ("CD", "UD"),
    "P": ("CP", "UP"),
    "W": ("CW", "UW"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare SDNET2018 weak YOLO labels and train an Ultralytics YOLO26 "
            "detector. SDNET2018 has image-level crack labels only, so cracked "
            "patches receive one full-image box and uncracked patches receive "
            "an empty label file."
        )
    )
    parser.add_argument("--sdnet-root", type=Path, default=Path("SDNET2018"), help="Path to the SDNET2018 root folder.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/sdnet2018_yolo_weak"),
        help="Output Ultralytics dataset directory.",
    )
    parser.add_argument("--model", default="yolo26n.pt", help="Ultralytics YOLO26 model, e.g. yolo26n.pt or yolo26.yaml.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=256, help="Training image size.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", default=None, help="Training device, e.g. 0, 0,1, cpu. Default lets Ultralytics decide.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--project", default="runs/detect", help="Ultralytics project output directory.")
    parser.add_argument("--name", default="sdnet2018_yolo26", help="Ultralytics run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow Ultralytics to reuse an existing run directory.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the prepared YOLO dataset directory.")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare dataset files; do not start training.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    return parser.parse_args()


def scan_sdnet2018(root: Path) -> list[ImageSample]:
    if not root.exists():
        raise FileNotFoundError(f"SDNET2018 root does not exist: {root}")

    samples: list[ImageSample] = []
    for surface, label_dirs in LABEL_DIRS.items():
        for label_dir in label_dirs:
            folder = root / surface / label_dir
            if not folder.exists():
                continue

            cracked = label_dir.startswith("C")
            for image_path in sorted(folder.glob("*.jpg")):
                samples.append(
                    ImageSample(
                        image_path=image_path,
                        surface=surface,
                        label_dir=label_dir,
                        cracked=cracked,
                    )
                )

    if not samples:
        raise RuntimeError(f"No SDNET2018 JPG images found under {root}")
    return samples


def split_samples(samples: list[ImageSample], val_ratio: float, seed: int) -> tuple[list[ImageSample], list[ImageSample]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be between 0 and 1, got {val_ratio}")

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    return train_samples, val_samples


def prepare_ultralytics_dataset(
    samples: list[ImageSample],
    dataset_dir: Path,
    val_ratio: float,
    seed: int,
    rebuild: bool,
) -> Path:
    data_yaml = dataset_dir / "sdnet2018.yaml"
    if data_yaml.exists() and not rebuild:
        return data_yaml

    if dataset_dir.exists() and rebuild:
        shutil.rmtree(dataset_dir)

    train_samples, val_samples = split_samples(samples, val_ratio, seed)
    for split, split_samples_ in (("train", train_samples), ("val", val_samples)):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        progress = tqdm(split_samples_, desc=f"Preparing {split}", unit="image")
        for sample in progress:
            output_name = f"{sample.surface}_{sample.label_dir}_{sample.image_path.stem}{sample.image_path.suffix.lower()}"
            output_image = image_dir / output_name
            output_label = label_dir / f"{Path(output_name).stem}.txt"

            link_or_copy(sample.image_path, output_image)
            write_yolo_label(output_label, sample.cracked)

    data_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: crack",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_yolo_label(label_path: Path, cracked: bool) -> None:
    if cracked:
        label_path.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")
    else:
        label_path.write_text("", encoding="utf-8")


def configure_ultralytics_paths(workspace: Path) -> None:
    config_root = workspace / ".ultralytics"
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))


def train(args: argparse.Namespace, data_yaml: Path) -> None:
    configure_ultralytics_paths(Path.cwd())

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)


def main() -> None:
    args = parse_args()

    samples = scan_sdnet2018(args.sdnet_root)
    if args.limit is not None:
        samples = samples[: args.limit]

    data_yaml = prepare_ultralytics_dataset(
        samples=samples,
        dataset_dir=args.dataset_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        rebuild=args.rebuild,
    )

    cracked = sum(sample.cracked for sample in samples)
    print(f"Prepared dataset yaml: {data_yaml}")
    print(f"Samples: {len(samples)} | cracked: {cracked} | uncracked: {len(samples) - cracked}")
    print("Weak label rule: cracked -> full-image YOLO box, uncracked -> empty label file")

    if args.prepare_only:
        return

    train(args, data_yaml)


if __name__ == "__main__":
    main()
