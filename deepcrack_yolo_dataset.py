from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Callable, Iterable, Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


DeepCrackSplit = Literal["train", "test"]
BoxMode = Literal["mask", "components"]


@dataclass(frozen=True)
class DeepCrackSample:
    image_path: Path
    mask_path: Path
    split: DeepCrackSplit


class DeepCrackYoloDataset(Dataset):
    """DeepCrack segmentation masks exposed as YOLO-style detection targets.

    Expected DeepCrack layout:

    ``root/train_img/*.jpg``
    ``root/train_lab/*.png``
    ``root/test_img/*.jpg``
    ``root/test_lab/*.png``

    The mask foreground is converted to normalized YOLO ``[class, x, y, w, h]``
    boxes. By default, each image receives one box enclosing all crack pixels.
    Set ``box_mode="components"`` to create one box per connected foreground
    component.
    """

    SPLIT_DIRS: dict[DeepCrackSplit, tuple[str, str]] = {
        "train": ("train_img", "train_lab"),
        "test": ("test_img", "test_lab"),
    }

    def __init__(
        self,
        root: str | Path,
        *,
        splits: Iterable[DeepCrackSplit] = ("train",),
        box_mode: BoxMode = "mask",
        mask_threshold: int = 127,
        min_area: int = 1,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        return_meta: bool = True,
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> None:
        self.root = Path(root)
        self.box_mode = box_mode
        self.mask_threshold = mask_threshold
        self.min_area = min_area
        self.transform = transform
        self.target_transform = target_transform
        self.return_meta = return_meta
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)

        if not self.root.exists():
            raise FileNotFoundError(f"DeepCrack root does not exist: {self.root}")
        if self.box_mode not in ("mask", "components"):
            raise ValueError(f"box_mode must be 'mask' or 'components', got {box_mode!r}")
        if not 0 <= self.mask_threshold <= 255:
            raise ValueError(f"mask_threshold must be in [0, 255], got {mask_threshold}")
        if self.min_area < 1:
            raise ValueError(f"min_area must be >= 1, got {min_area}")

        self.samples = self._scan_samples(splits)
        if not self.samples:
            raise RuntimeError(
                f"No DeepCrack image/mask pairs found under {self.root}. "
                "Expected train_img/train_lab or test_img/test_lab directories."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
        with Image.open(sample.mask_path) as mask:
            mask = mask.convert("L")

        if image.size != mask.size:
            raise ValueError(
                f"Image and mask sizes differ for {sample.image_path}: "
                f"image={image.size}, mask={mask.size}"
            )

        target = self._target_from_mask(mask)

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
            "mask_path": str(sample.mask_path),
            "split": sample.split,
            "box_mode": self.box_mode,
        }
        return image, target, meta

    def _scan_samples(self, splits: Iterable[DeepCrackSplit]) -> list[DeepCrackSample]:
        samples: list[DeepCrackSample] = []

        for split in splits:
            if split not in self.SPLIT_DIRS:
                valid = ", ".join(self.SPLIT_DIRS)
                raise ValueError(f"Unknown split '{split}'. Valid splits: {valid}")

            image_dir_name, mask_dir_name = self.SPLIT_DIRS[split]
            image_dir = self.root / image_dir_name
            mask_dir = self.root / mask_dir_name
            if not image_dir.exists() or not mask_dir.exists():
                continue

            for image_path in sorted(image_dir.iterdir()):
                if (
                    not image_path.is_file()
                    or image_path.name.startswith("._")
                    or image_path.suffix.lower() not in self.image_extensions
                ):
                    continue

                mask_path = mask_dir / f"{image_path.stem}.png"
                if not mask_path.exists():
                    raise FileNotFoundError(f"Missing DeepCrack mask for {image_path}: {mask_path}")

                samples.append(
                    DeepCrackSample(
                        image_path=image_path,
                        mask_path=mask_path,
                        split=split,
                    )
                )

        return samples

    def _target_from_mask(self, mask: Image.Image) -> torch.Tensor:
        foreground = self._mask_to_foreground(mask)
        if not foreground.any():
            return torch.empty((0, 5), dtype=torch.float32)

        if self.box_mode == "components":
            boxes = self._component_boxes(foreground, mask.width, mask.height)
        else:
            boxes = [self._foreground_box(foreground, mask.width, mask.height)]

        if not boxes:
            return torch.empty((0, 5), dtype=torch.float32)

        return torch.tensor(boxes, dtype=torch.float32)

    def _mask_to_foreground(self, mask: Image.Image) -> torch.Tensor:
        data = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8)
        data = data.view(mask.height, mask.width)
        return data > self.mask_threshold

    @staticmethod
    def _foreground_box(foreground: torch.Tensor, width: int, height: int) -> list[float]:
        ys, xs = torch.nonzero(foreground, as_tuple=True)
        return DeepCrackYoloDataset._to_yolo_box(
            x_min=int(xs.min()),
            y_min=int(ys.min()),
            x_max=int(xs.max()) + 1,
            y_max=int(ys.max()) + 1,
            image_width=width,
            image_height=height,
        )

    def _component_boxes(self, foreground: torch.Tensor, width: int, height: int) -> list[list[float]]:
        visited = torch.zeros_like(foreground, dtype=torch.bool)
        boxes: list[list[float]] = []

        for y0, x0 in torch.nonzero(foreground, as_tuple=False).tolist():
            if visited[y0, x0]:
                continue

            area = 0
            x_min = x_max = x0
            y_min = y_max = y0
            queue: deque[tuple[int, int]] = deque([(y0, x0)])
            visited[y0, x0] = True

            while queue:
                y, x = queue.popleft()
                area += 1
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

                for ny, nx in (
                    (y - 1, x),
                    (y + 1, x),
                    (y, x - 1),
                    (y, x + 1),
                    (y - 1, x - 1),
                    (y - 1, x + 1),
                    (y + 1, x - 1),
                    (y + 1, x + 1),
                ):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and foreground[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if area >= self.min_area:
                boxes.append(
                    self._to_yolo_box(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max + 1,
                        y_max=y_max + 1,
                        image_width=width,
                        image_height=height,
                    )
                )

        return boxes

    @staticmethod
    def _to_yolo_box(
        *,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        image_width: int,
        image_height: int,
    ) -> list[float]:
        box_width = max(1, x_max - x_min)
        box_height = max(1, y_max - y_min)
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
    """Create an Ultralytics YOLO dataset from DeepCrack and return data.yaml."""

    source_root = Path(path)
    output_root = Path("datasets") / "deepcrack_yolo"
    data_yaml = output_root / "data.yaml"
    if data_yaml.exists():
        return data_yaml

    for split, source_split in (("train", "train"), ("val", "test")):
        dataset = DeepCrackYoloDataset(source_root, splits=(source_split,), return_meta=True)
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        progress = tqdm(dataset.samples, desc=f"Preparing {split}", unit="image")
        for sample in progress:
            image_name = sample.image_path.name
            output_image = image_dir / image_name
            output_label = label_dir / f"{sample.image_path.stem}.txt"

            link_or_copy(sample.image_path, output_image)
            with Image.open(sample.mask_path) as mask:
                target = dataset._target_from_mask(mask.convert("L"))
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
    dataset = DeepCrackYoloDataset("Deepcrack/DeepCrack-master/dataset/DeepCrack", splits=("train", "test"))
    image, target, meta = dataset[0]

    split_counts = {split: 0 for split in dataset.SPLIT_DIRS}
    for sample in dataset.samples:
        split_counts[sample.split] += 1

    print(f"samples: {len(dataset)}")
    print(f"train: {split_counts['train']}")
    print(f"test: {split_counts['test']}")
    print(f"image shape: {tuple(image.shape)}")
    print(f"target: {target.tolist()}")
    print(f"meta: {meta}")
