from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Literal

from PIL import Image
import torch
from torch.utils.data import Dataset

from dataset.dacl_yolo_dataset import DACL_CLASSES, CLASS_TO_ID


DaclSplit = Literal["train", "validation"]

DACL_SEGMENTATION_CLASS_NAMES = ["background", *DACL_CLASSES]


@dataclass(frozen=True)
class DaclSegmentationSample:
    image_path: Path
    annotation_path: Path
    split: DaclSplit


class DaclBboxSegmentationDataset(Dataset):
    """DACL polygon annotations converted to bbox-filled semantic masks.

    DACL polygons are first converted to their minimum enclosing bbox. All pixels
    inside that bbox are filled with the corresponding class id. Class ``0`` is
    background, so DACL class ids are shifted by ``+1`` in the mask.
    """

    CLASSES = DACL_SEGMENTATION_CLASS_NAMES
    DACL_CLASSES = DACL_CLASSES
    CLASS_TO_ID = CLASS_TO_ID
    SPLIT_DIRS: dict[DaclSplit, tuple[str, str]] = {
        "train": ("train", "train"),
        "validation": ("validation", "validation"),
    }
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        root: str | Path,
        *,
        split: DaclSplit = "train",
        image_size: int | tuple[int, int] = (512, 512),
        return_meta: bool = False,
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = normalize_image_size(image_size)
        self.return_meta = return_meta
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)
        self.class_names = list(self.CLASSES)
        self.num_classes = len(self.class_names)

        if not self.root.exists():
            raise FileNotFoundError(f"DACL root does not exist: {self.root}")
        if self.split not in self.SPLIT_DIRS:
            valid = ", ".join(self.SPLIT_DIRS)
            raise ValueError(f"Unknown split '{split}'. Valid splits: {valid}")

        self.samples = self._scan_samples(split)
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
            image_width, image_height = image.size
            image = image.resize((self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR)

        annotation = self._load_annotation(sample.annotation_path)
        mask = self._mask_from_annotation(
            annotation,
            image_width=image_width,
            image_height=image_height,
        )
        mask_image = Image.fromarray(mask.byte().numpy(), mode="L")
        mask_image = mask_image.resize((self.image_size[1], self.image_size[0]), Image.Resampling.NEAREST)

        image_tensor = self._pil_to_normalized_tensor(image)
        mask_tensor = torch.frombuffer(bytearray(mask_image.tobytes()), dtype=torch.uint8)
        mask_tensor = mask_tensor.view(self.image_size[0], self.image_size[1]).long()

        if not self.return_meta:
            return image_tensor, mask_tensor

        meta = {
            "path": str(sample.image_path),
            "annotation_path": str(sample.annotation_path),
            "split": sample.split,
            "classes": self.class_names,
        }
        return image_tensor, mask_tensor, meta

    def _scan_samples(self, split: DaclSplit) -> list[DaclSegmentationSample]:
        image_split, annotation_split = self.SPLIT_DIRS[split]
        image_dir = self.root / "train_phase" / "images" / image_split
        annotation_dir = self.root / "train_phase" / "annotations" / annotation_split
        if not image_dir.exists() or not annotation_dir.exists():
            return []

        samples: list[DaclSegmentationSample] = []
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
                DaclSegmentationSample(
                    image_path=image_path,
                    annotation_path=annotation_path,
                    split=split,
                )
            )
        return samples

    @staticmethod
    def _load_annotation(annotation_path: Path) -> dict:
        return json.loads(annotation_path.read_text(encoding="utf-8"))

    def _mask_from_annotation(self, annotation: dict, *, image_width: int, image_height: int) -> torch.Tensor:
        mask = torch.zeros((image_height, image_width), dtype=torch.uint8)

        for shape in annotation.get("shapes", []):
            label = shape.get("label")
            points = shape.get("points", [])
            if label not in self.CLASS_TO_ID or len(points) < 2:
                continue

            bbox = points_to_bbox(points, image_width=image_width, image_height=image_height)
            if bbox is None:
                continue

            x_min, y_min, x_max, y_max = bbox
            mask[y_min:y_max, x_min:x_max] = self.CLASS_TO_ID[label] + 1

        return mask

    @classmethod
    def _pil_to_normalized_tensor(cls, image: Image.Image) -> torch.Tensor:
        data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        data = data.view(image.height, image.width, len(image.getbands()))
        tensor = data.permute(2, 0, 1).contiguous().float().div(255.0)
        mean = torch.tensor(cls.MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(cls.STD, dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std


def points_to_bbox(
    points: list[list[float]],
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]

    x_min = max(0, min(image_width - 1, math.floor(min(xs))))
    y_min = max(0, min(image_height - 1, math.floor(min(ys))))
    x_max = max(0, min(image_width, math.ceil(max(xs))))
    y_max = max(0, min(image_height, math.ceil(max(ys))))

    if x_max <= x_min:
        x_max = min(image_width, x_min + 1)
    if y_max <= y_min:
        y_max = min(image_height, y_min + 1)
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def normalize_image_size(image_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        return image_size, image_size
    if len(image_size) != 2:
        raise ValueError(f"image_size must be int or (height, width), got {image_size}")
    height, width = int(image_size[0]), int(image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size values must be positive, got {image_size}")
    return height, width


def segmentation_collate_fn(batch):
    if len(batch[0]) == 2:
        images, masks = zip(*batch)
        return torch.stack(images, dim=0), torch.stack(masks, dim=0)

    images, masks, metas = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(masks, dim=0), list(metas)


def create_dataset(
    path: str | Path,
    *,
    split: DaclSplit = "train",
    image_size: int | tuple[int, int] = (512, 512),
    return_meta: bool = False,
) -> tuple[DaclBboxSegmentationDataset, int]:
    dataset = DaclBboxSegmentationDataset(
        path,
        split=split,
        image_size=image_size,
        return_meta=return_meta,
    )
    return dataset, dataset.num_classes


def create_train_val_datasets(
    path: str | Path,
    *,
    image_size: int | tuple[int, int] = (512, 512),
    return_meta: bool = False,
) -> tuple[DaclBboxSegmentationDataset, DaclBboxSegmentationDataset, int]:
    train_dataset, num_classes = create_dataset(
        path,
        split="train",
        image_size=image_size,
        return_meta=return_meta,
    )
    val_dataset, val_num_classes = create_dataset(
        path,
        split="validation",
        image_size=image_size,
        return_meta=return_meta,
    )
    if val_num_classes != num_classes:
        raise ValueError(f"Class count mismatch: train={num_classes}, validation={val_num_classes}")
    return train_dataset, val_dataset, num_classes


if __name__ == "__main__":
    dataset, class_num = create_dataset("RawDataset/dacl", split="train")
    image, mask = dataset[0]
    print(f"samples: {len(dataset)}")
    print(f"classes: {class_num} {dataset.class_names}")
    print(f"image shape: {tuple(image.shape)}")
    print(f"mask shape: {tuple(mask.shape)}")
    print(f"mask labels: {mask.unique().tolist()}")
