from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
import shutil
from typing import Callable, Iterable, Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


SurfaceType = Literal["D", "P", "W"]


@dataclass(frozen=True)
class SDNET2018Sample:
    image_path: Path
    surface: SurfaceType
    label_dir: str
    cracked: bool


class SDNET2018YoloDataset(Dataset):
    """SDNET2018 as a YOLO-style detection dataset.

    SDNET2018 provides image-level labels for 256 x 256 cropped concrete
    patches, not pixel masks or crack bounding boxes. For detection pipelines,
    this dataset exposes a weak YOLO target:

    - cracked patch: one box covering the full image, ``[class, x, y, w, h]``
      where coordinates are normalized YOLO xywh values.
    - uncracked patch: an empty ``(0, 5)`` target tensor.

    Directory layout expected by this class:

    ``root/D/CD/*.jpg``  cracked deck
    ``root/D/UD/*.jpg``  uncracked deck
    ``root/P/CP/*.jpg``  cracked pavement
    ``root/P/UP/*.jpg``  uncracked pavement
    ``root/W/CW/*.jpg``  cracked wall
    ``root/W/UW/*.jpg``  uncracked wall
    """

    SURFACE_NAMES = {
        "D": "deck",
        "P": "pavement",
        "W": "wall",
    }
    LABEL_DIRS = {
        "D": ("CD", "UD"),
        "P": ("CP", "UP"),
        "W": ("CW", "UW"),
    }

    def __init__(
        self,
        root: str | Path,
        *,
        surfaces: Iterable[SurfaceType] = ("D", "P", "W"),
        cracked_only: bool = False,
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
            raise FileNotFoundError(f"SDNET2018 root does not exist: {self.root}")

        self.samples = self._scan_samples(surfaces, cracked_only)
        if not self.samples:
            raise RuntimeError(
                f"No SDNET2018 images found under {self.root}. "
                "Expected directories like D/CD, D/UD, P/CP, P/UP, W/CW, W/UW."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")

        target = self._target_for_sample(sample)

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
            "surface": sample.surface,
            "surface_name": self.SURFACE_NAMES[sample.surface],
            "label_dir": sample.label_dir,
            "cracked": sample.cracked,
        }
        return image, target, meta

    def _scan_samples(
        self,
        surfaces: Iterable[SurfaceType],
        cracked_only: bool,
    ) -> list[SDNET2018Sample]:
        samples: list[SDNET2018Sample] = []

        for surface in surfaces:
            if surface not in self.LABEL_DIRS:
                valid = ", ".join(self.LABEL_DIRS)
                raise ValueError(f"Unknown surface '{surface}'. Valid surfaces: {valid}")

            cracked_dir, uncracked_dir = self.LABEL_DIRS[surface]
            label_dirs = (cracked_dir,) if cracked_only else (cracked_dir, uncracked_dir)

            for label_dir in label_dirs:
                folder = self.root / surface / label_dir
                if not folder.exists():
                    continue

                cracked = label_dir.startswith("C")
                for image_path in sorted(folder.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in self.image_extensions:
                        samples.append(
                            SDNET2018Sample(
                                image_path=image_path,
                                surface=surface,
                                label_dir=label_dir,
                                cracked=cracked,
                            )
                        )

        return samples

    @staticmethod
    def _target_for_sample(sample: SDNET2018Sample) -> torch.Tensor:
        if not sample.cracked:
            return torch.empty((0, 5), dtype=torch.float32)

        # class_id=0 means "crack"; normalized xywh covers the full 256x256 patch.
        return torch.tensor([[0.0, 0.5, 0.5, 1.0, 1.0]], dtype=torch.float32)

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
    """Create a weak Ultralytics YOLO dataset from SDNET2018 and return data.yaml."""

    source_root = Path(path)
    output_root = Path("datasets") / "sdnet2018_yolo_weak"
    data_yaml = output_root / "data.yaml"
    if data_yaml.exists():
        return data_yaml

    dataset = SDNET2018YoloDataset(source_root, return_meta=True)
    train_samples, val_samples = split_samples(dataset.samples, val_ratio=0.2, seed=42)

    for split, samples in (("train", train_samples), ("val", val_samples)):
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        progress = tqdm(samples, desc=f"Preparing {split}", unit="image")
        for sample in progress:
            output_name = f"{sample.surface}_{sample.label_dir}_{sample.image_path.stem}{sample.image_path.suffix.lower()}"
            output_image = image_dir / output_name
            output_label = label_dir / f"{Path(output_name).stem}.txt"

            link_or_copy(sample.image_path, output_image)
            write_yolo_label(output_label, SDNET2018YoloDataset._target_for_sample(sample))

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


def split_samples(
    samples: list[SDNET2018Sample],
    val_ratio: float,
    seed: int,
) -> tuple[list[SDNET2018Sample], list[SDNET2018Sample]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    return shuffled[val_count:], shuffled[:val_count]


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
    dataset = SDNET2018YoloDataset("SDNET2018")
    image, target, meta = dataset[0]

    cracked_count = sum(sample.cracked for sample in dataset.samples)
    uncracked_count = len(dataset) - cracked_count

    print(f"samples: {len(dataset)}")
    print(f"cracked: {cracked_count}")
    print(f"uncracked: {uncracked_count}")
    print(f"image shape: {tuple(image.shape)}")
    print(f"target: {target.tolist()}")
    print(f"meta: {meta}")
