from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Callable, Iterable, Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


CSBSplit = Literal["train", "test"]
REBUILD_DATASET = True


@dataclass(frozen=True)
class CSBSample:
    image_path: Path
    annotation_path: Path
    cracked: bool
    split: CSBSplit


class CSBYoloDataset(Dataset):
    """CSB crack annotations exposed as YOLO-style detection targets.

    Expected default layout:

    ``root/entire images/crack_train/*.JPG``
    ``root/entire images/nocrack_train/*.JPG``
    ``root/entire images/crack_test/*.JPG``
    ``root/entire images/nocrack_test/*.JPG``

    Each image has a JSON file with ``annotations.crack_pixels``. The crack
    pixels are stored as ``[row, col]`` pairs, so they are converted to a single
    minimum bounding box per image in normalized YOLO ``[class, x, y, w, h]``
    format. Images with no crack pixels return an empty ``(0, 5)`` tensor.
    """

    CLASSES = ["crack"]
    SPLIT_DIRS = {
        "train": ("crack_train", "nocrack_train"),
        "test": ("crack_test", "nocrack_test"),
    }

    def __init__(
        self,
        root: str | Path,
        *,
        splits: Iterable[CSBSplit] = ("train",),
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
            raise FileNotFoundError(f"CSB root does not exist: {self.root}")

        self.samples = self._scan_samples(splits)
        if not self.samples:
            raise RuntimeError(
                f"No CSB image/annotation pairs found under {self.root}. "
                "Expected entire images/crack_train, nocrack_train, crack_test, nocrack_test."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            image_width, image_height = image.size

        annotation = self._load_annotation(sample.annotation_path)
        target = self._target_from_annotation(
            annotation,
            image_width=image_width,
            image_height=image_height,
            cracked=sample.cracked,
        )

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
            "annotation_path": str(sample.annotation_path),
            "split": sample.split,
            "cracked": sample.cracked,
        }
        return image, target, meta

    def _scan_samples(self, splits: Iterable[CSBSplit]) -> list[CSBSample]:
        samples: list[CSBSample] = []
        base_dir = self.root / "entire images"

        for split in splits:
            if split not in self.SPLIT_DIRS:
                valid = ", ".join(self.SPLIT_DIRS)
                raise ValueError(f"Unknown split '{split}'. Valid splits: {valid}")

            crack_dir, nocrack_dir = self.SPLIT_DIRS[split]
            for folder_name, cracked in ((crack_dir, True), (nocrack_dir, False)):
                folder = base_dir / folder_name
                if not folder.exists():
                    continue

                for image_path in sorted(folder.iterdir()):
                    if (
                        not image_path.is_file()
                        or image_path.name.startswith("._")
                        or image_path.suffix.lower() not in self.image_extensions
                    ):
                        continue

                    annotation_path = image_path.with_suffix(".json")
                    if not annotation_path.exists():
                        raise FileNotFoundError(f"Missing CSB annotation for {image_path}: {annotation_path}")

                    samples.append(
                        CSBSample(
                            image_path=image_path,
                            annotation_path=annotation_path,
                            cracked=cracked,
                            split=split,
                        )
                    )

        return samples

    @staticmethod
    def _load_annotation(annotation_path: Path) -> dict:
        return json.loads(annotation_path.read_text(encoding="utf-8"))

    def _target_from_annotation(
        self,
        annotation: dict,
        *,
        image_width: int,
        image_height: int,
        cracked: bool = False,
    ) -> torch.Tensor:
        crack_pixels = annotation.get("annotations", {}).get("crack_pixels", [])
        if not crack_pixels and cracked:
            return torch.tensor([[0.0, 0.5, 0.5, 1.0, 1.0]], dtype=torch.float32)
        if not crack_pixels:
            return torch.empty((0, 5), dtype=torch.float32)

        box = self._crack_pixels_to_yolo_box(
            crack_pixels,
            image_width=image_width,
            image_height=image_height,
        )
        if box is None:
            return torch.empty((0, 5), dtype=torch.float32)
        return torch.tensor([box], dtype=torch.float32)

    @staticmethod
    def _crack_pixels_to_yolo_box(
        crack_pixels: list[list[int | float]],
        *,
        image_width: int,
        image_height: int,
    ) -> list[float] | None:
        rows = [float(pixel[0]) for pixel in crack_pixels]
        cols = [float(pixel[1]) for pixel in crack_pixels]
        if not rows or not cols:
            return None

        x_min = max(0.0, min(cols))
        y_min = max(0.0, min(rows))
        x_max = min(float(image_width), max(cols) + 1.0)
        y_max = min(float(image_height), max(rows) + 1.0)

        box_width = x_max - x_min
        box_height = y_max - y_min
        if box_width <= 0.0 or box_height <= 0.0:
            return None

        x_center = x_min + box_width / 2.0
        y_center = y_min + box_height / 2.0
        return [
            0.0,
            x_center / image_width,
            y_center / image_height,
            box_width / image_width,
            box_height / image_height,
        ]

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
    """Create an Ultralytics YOLO dataset from CSB and return data.yaml."""

    source_root = Path(path)
    output_root = Path("datasets") / "csb_yolo"
    data_yaml = output_root / "data.yaml"
    if data_yaml.exists() and not REBUILD_DATASET:
        return data_yaml
    if output_root.exists() and REBUILD_DATASET:
        shutil.rmtree(output_root)

    for output_split, source_split in (("train", "train"), ("val", "test")):
        dataset = CSBYoloDataset(source_root, splits=(source_split,), return_meta=True)
        image_dir = output_root / "images" / output_split
        label_dir = output_root / "labels" / output_split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        progress = tqdm(dataset.samples, desc=f"Preparing {output_split}", unit="image")
        for sample in progress:
            output_image = image_dir / sample.image_path.name
            output_label = label_dir / f"{sample.image_path.stem}.txt"

            link_or_copy(sample.image_path, output_image)
            with Image.open(sample.image_path) as image:
                image_width, image_height = image.size
            annotation = dataset._load_annotation(sample.annotation_path)
            target = dataset._target_from_annotation(
                annotation,
                image_width=image_width,
                image_height=image_height,
                cracked=sample.cracked,
            )
            write_yolo_label(output_label, target)

    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve().as_posix()}",
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
    dataset = CSBYoloDataset("RawDataset/CSB_dataset", splits=("train", "test"))
    image, target, meta = dataset[0]

    split_counts = {split: 0 for split in dataset.SPLIT_DIRS}
    cracked_count = 0
    for sample in dataset.samples:
        split_counts[sample.split] += 1
        cracked_count += int(sample.cracked)

    print(f"samples: {len(dataset)}")
    print(f"train: {split_counts['train']}")
    print(f"test: {split_counts['test']}")
    print(f"cracked folders: {cracked_count}")
    print(f"image shape: {tuple(image.shape)}")
    print(f"target: {target.tolist()}")
    print(f"meta: {meta}")
