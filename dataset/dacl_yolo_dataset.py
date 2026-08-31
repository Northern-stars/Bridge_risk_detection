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


DaclSplit = Literal["train", "validation"]


DACL_CLASSES = [
    "Crack",
    "ACrack",
    "Wetspot",
    "Efflorescence",
    "Rust",
    "Rockpocket",
    "Hollowareas",
    "Cavity",
    "Spalling",
    "Graffiti",
    "Weathering",
    "Restformwork",
    "ExposedRebars",
    "Bearing",
    "EJoint",
    "Drainage",
    "PEquipment",
    "JTape",
    "WConccor",
]
CLASS_TO_ID = {name: index for index, name in enumerate(DACL_CLASSES)}


@dataclass(frozen=True)
class DaclSample:
    image_path: Path
    annotation_path: Path
    split: DaclSplit


class DaclYoloDataset(Dataset):
    """DACL10K polygon annotations exposed as YOLO-style detection targets.

    Expected DACL layout:

    ``root/train_phase/images/train/*.jpg``
    ``root/train_phase/images/validation/*.jpg``
    ``root/train_phase/annotations/train/*.json``
    ``root/train_phase/annotations/validation/*.json``

    DACL10K is a multi-label semantic segmentation dataset. This dataset turns
    each annotated polygon into one normalized YOLO ``[class, x, y, w, h]`` box.
    Images with no shapes return an empty ``(0, 5)`` target tensor.
    """

    CLASSES = DACL_CLASSES
    CLASS_TO_ID = CLASS_TO_ID
    SPLIT_DIRS: dict[DaclSplit, tuple[str, str]] = {
        "train": ("train", "train"),
        "validation": ("validation", "validation"),
    }

    def __init__(
        self,
        root: str | Path,
        *,
        splits: Iterable[DaclSplit] = ("train",),
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        return_meta: bool = True,
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.return_meta = return_meta
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)

        if not self.root.exists():
            raise FileNotFoundError(f"DACL root does not exist: {self.root}")

        self.samples = self._scan_samples(splits)
        if not self.samples:
            raise RuntimeError(
                f"No DACL image/annotation pairs found under {self.root}. "
                "Expected train_phase/images/{train,validation} and "
                "train_phase/annotations/{train,validation}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")

        annotation = self._load_annotation(sample.annotation_path)
        target = self._target_from_annotation(annotation)

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
            "classes": self.CLASSES,
        }
        return image, target, meta

    def _scan_samples(self, splits: Iterable[DaclSplit]) -> list[DaclSample]:
        samples: list[DaclSample] = []

        for split in splits:
            if split not in self.SPLIT_DIRS:
                valid = ", ".join(self.SPLIT_DIRS)
                raise ValueError(f"Unknown split '{split}'. Valid splits: {valid}")

            image_split, annotation_split = self.SPLIT_DIRS[split]
            image_dir = self.root / "train_phase" / "images" / image_split
            annotation_dir = self.root / "train_phase" / "annotations" / annotation_split
            if not image_dir.exists() or not annotation_dir.exists():
                continue

            for image_path in sorted(image_dir.iterdir()):
                if (
                    not image_path.is_file()
                    or image_path.name.startswith("._")
                    or image_path.suffix.lower() not in self.image_extensions
                ):
                    continue

                annotation_path = annotation_dir / f"{image_path.stem}.json"
                if not annotation_path.exists():
                    raise FileNotFoundError(f"Missing DACL annotation for {image_path}: {annotation_path}")

                samples.append(
                    DaclSample(
                        image_path=image_path,
                        annotation_path=annotation_path,
                        split=split,
                    )
                )

        return samples

    @staticmethod
    def _load_annotation(annotation_path: Path) -> dict:
        return json.loads(annotation_path.read_text(encoding="utf-8"))

    def _target_from_annotation(self, annotation: dict) -> torch.Tensor:
        image_width = int(annotation["imageWidth"])
        image_height = int(annotation["imageHeight"])
        boxes: list[list[float]] = []

        for shape in annotation.get("shapes", []):
            label = shape.get("label")
            points = shape.get("points", [])
            if label not in self.CLASS_TO_ID or len(points) < 2:
                continue

            box = self._points_to_yolo_box(
                class_id=self.CLASS_TO_ID[label],
                points=points,
                image_width=image_width,
                image_height=image_height,
            )
            if box is not None:
                boxes.append(box)

        if not boxes:
            return torch.empty((0, 5), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

    @staticmethod
    def _points_to_yolo_box(
        *,
        class_id: int,
        points: list[list[float]],
        image_width: int,
        image_height: int,
    ) -> list[float] | None:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]

        x_min = max(0.0, min(xs))
        y_min = max(0.0, min(ys))
        x_max = min(float(image_width), max(xs))
        y_max = min(float(image_height), max(ys))

        box_width = x_max - x_min
        box_height = y_max - y_min
        if box_width <= 0.0 or box_height <= 0.0:
            return None

        x_center = x_min + box_width / 2.0
        y_center = y_min + box_height / 2.0
        return [
            float(class_id),
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
    """Create an Ultralytics YOLO dataset from DACL10K and return data.yaml."""

    source_root = Path(path)
    output_root = Path("datasets") / "dacl_yolo"
    data_yaml = output_root / "data.yaml"
    if data_yaml.exists():
        return data_yaml

    for split, source_split in (("train", "train"), ("val", "validation")):
        dataset = DaclYoloDataset(source_root, splits=(source_split,), return_meta=True)
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        progress = tqdm(dataset.samples, desc=f"Preparing {split}", unit="image")
        for sample in progress:
            output_image = image_dir / sample.image_path.name
            output_label = label_dir / f"{sample.image_path.stem}.txt"

            link_or_copy(sample.image_path, output_image)
            annotation = dataset._load_annotation(sample.annotation_path)
            target = dataset._target_from_annotation(annotation)
            write_yolo_label(output_label, target)

    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(DACL_CLASSES)],
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
    dataset = DaclYoloDataset("RawDataset/dacl", splits=("train", "validation"))
    image, target, meta = dataset[0]

    split_counts = {split: 0 for split in dataset.SPLIT_DIRS}
    for sample in dataset.samples:
        split_counts[sample.split] += 1

    print(f"samples: {len(dataset)}")
    print(f"train: {split_counts['train']}")
    print(f"validation: {split_counts['validation']}")
    print(f"classes: {len(dataset.CLASSES)}")
    print(f"image shape: {tuple(image.shape)}")
    print(f"target shape: {tuple(target.shape)}")
    print(f"meta: {meta}")
