from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Callable, Iterable, Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


MultiDefactSplit = Literal["train", "valid", "test"]


MULTI_DEFACT_CLASSES = [
    "Cracks",
    "Spalling",
    "Honeycomb_Surface",
    "Exposed_Rebar",
    "Seepage",
    "Hole",
]


@dataclass(frozen=True)
class MultiDefactSample:
    image_path: Path
    label_path: Path | None
    split: MultiDefactSplit


class MultiDefactYoloDataset(Dataset):
    """Multi-defact YOLO labels exposed through the common dataset interface.

    Expected layout:

    ``root/train/images/*``
    ``root/train/labels/*.txt``
    ``root/valid/images/*``
    ``root/valid/labels/*.txt``
    ``root/test/images/*``
    ``root/test/labels/*.txt``

    The label files are already in YOLO detection format:
    ``class_id x_center y_center width height`` with normalized coordinates.
    Images without a matching label file are kept as empty-label samples.
    Label files without a matching image are ignored.
    """

    CLASSES = MULTI_DEFACT_CLASSES
    SPLITS: tuple[MultiDefactSplit, ...] = ("train", "valid", "test")

    def __init__(
        self,
        root: str | Path,
        *,
        splits: Iterable[MultiDefactSplit] = ("train",),
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        return_meta: bool = True,
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.return_meta = return_meta
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)

        if not self.root.exists():
            raise FileNotFoundError(f"multi_defact root does not exist: {self.root}")

        self.samples = self._scan_samples(splits)
        if not self.samples:
            raise RuntimeError(
                f"No multi_defact images found under {self.root}. "
                "Expected split folders with images/ and labels/."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")

        target = self._load_yolo_label(sample.label_path)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self._pil_to_tensor(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.return_meta:
            return image, target

        meta = {
            "path": str(sample.image_path),
            "label_path": str(sample.label_path) if sample.label_path is not None else None,
            "split": sample.split,
            "classes": self.CLASSES,
        }
        return image, target, meta

    def _scan_samples(self, splits: Iterable[MultiDefactSplit]) -> list[MultiDefactSample]:
        samples: list[MultiDefactSample] = []

        for split in splits:
            if split not in self.SPLITS:
                valid = ", ".join(self.SPLITS)
                raise ValueError(f"Unknown split '{split}'. Valid splits: {valid}")

            image_dir = self.root / split / "images"
            label_dir = self.root / split / "labels"
            if not image_dir.exists():
                continue

            for image_path in sorted(image_dir.iterdir()):
                if (
                    not image_path.is_file()
                    or image_path.name.startswith("._")
                    or image_path.suffix.lower() not in self.image_extensions
                ):
                    continue

                label_path = label_dir / f"{image_path.stem}.txt"
                samples.append(
                    MultiDefactSample(
                        image_path=image_path,
                        label_path=label_path if label_path.exists() else None,
                        split=split,
                    )
                )

        return samples

    @staticmethod
    def _load_yolo_label(label_path: Path | None) -> torch.Tensor:
        if label_path is None:
            return torch.empty((0, 5), dtype=torch.float32)

        boxes: list[list[float]] = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}: {line!r}")

            class_id, x_center, y_center, width, height = (float(value) for value in parts)
            boxes.append([class_id, x_center, y_center, width, height])

        if not boxes:
            return torch.empty((0, 5), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        data = data.view(image.height, image.width, len(image.getbands()))
        return data.permute(2, 0, 1).contiguous().float().div(255.0)


def yolo_collate_fn(batch):
    """Collate variable-length YOLO targets for torch DataLoader."""

    if len(batch[0]) == 2:
        images, targets = zip(*batch)
        return torch.stack(images, dim=0), list(targets)

    images, targets, metas = zip(*batch)
    return torch.stack(images, dim=0), list(targets), list(metas)


def create_dataset(path: str | Path) -> Path:
    """Create an Ultralytics YOLO dataset from multi_defact and return data.yaml."""

    source_root = Path(path)
    output_root = Path("datasets") / "multi_defact_yolo"
    data_yaml = output_root / "data.yaml"
    if data_yaml.exists():
        return data_yaml

    split_map = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }
    for source_split, output_split in split_map.items():
        dataset = MultiDefactYoloDataset(source_root, splits=(source_split,), return_meta=True)
        image_dir = output_root / "images" / output_split
        label_dir = output_root / "labels" / output_split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        progress = tqdm(dataset.samples, desc=f"Preparing {output_split}", unit="image")
        for sample in progress:
            output_image = image_dir / sample.image_path.name
            output_label = label_dir / f"{sample.image_path.stem}.txt"

            link_or_copy(sample.image_path, output_image)
            target = dataset._load_yolo_label(sample.label_path)
            write_yolo_label(output_label, target)

    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(MULTI_DEFACT_CLASSES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_yolo_label(label_path: Path, target: torch.Tensor) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(f"{value:g}" for value in row.tolist()) for row in target]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


if __name__ == "__main__":
    dataset = MultiDefactYoloDataset("RawDataset/multi_defact", splits=("train", "valid", "test"))
    image, target, meta = dataset[0]

    split_counts = {split: 0 for split in dataset.SPLITS}
    missing_labels = 0
    for sample in dataset.samples:
        split_counts[sample.split] += 1
        if sample.label_path is None:
            missing_labels += 1

    print(f"samples: {len(dataset)}")
    print(f"train: {split_counts['train']}")
    print(f"valid: {split_counts['valid']}")
    print(f"test: {split_counts['test']}")
    print(f"missing labels: {missing_labels}")
    print(f"classes: {len(dataset.CLASSES)}")
    print(f"image shape: {tuple(image.shape)}")
    print(f"target shape: {tuple(target.shape)}")
    print(f"meta: {meta}")
