from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable, Iterable, Literal

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset


SurfaceType = Literal["D", "P", "W"]
ClassMode = Literal["binary", "surface_damage"]


@dataclass(frozen=True)
class SDNET2018Sample:
    image_path: Path
    surface: SurfaceType
    label_dir: str
    cracked: bool
    class_id: int


class SDNET2018ClassificationDataset(Dataset):
    """SDNET2018 as an image classification dataset.

    SDNET2018 is natively labeled at image level: each 256 x 256 RGB patch is
    either cracked or uncracked. The default ``class_mode="binary"`` uses:

    - ``0``: uncracked
    - ``1``: crack

    ``class_mode="surface_damage"`` exposes six classes by combining surface
    type and crack state: CD, UD, CP, UP, CW, UW.
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
    CLASS_NAMES_BY_MODE = {
        "binary": ("uncracked", "crack"),
        "surface_damage": ("CD", "UD", "CP", "UP", "CW", "UW"),
    }

    def __init__(
        self,
        root: str | Path,
        *,
        surfaces: Iterable[SurfaceType] = ("D", "P", "W"),
        class_mode: ClassMode = "binary",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        return_meta: bool = True,
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> None:
        self.root = Path(root)
        self.class_mode = class_mode
        self.transform = transform
        self.target_transform = target_transform
        self.return_meta = return_meta
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)

        if not self.root.exists():
            raise FileNotFoundError(f"SDNET2018 root does not exist: {self.root}")
        if self.class_mode not in self.CLASS_NAMES_BY_MODE:
            valid = ", ".join(self.CLASS_NAMES_BY_MODE)
            raise ValueError(f"Unknown class_mode '{class_mode}'. Valid modes: {valid}")

        self.class_names = list(self.CLASS_NAMES_BY_MODE[self.class_mode])
        self.num_classes = len(self.class_names)
        self.samples = self._scan_samples(surfaces)
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

        target = torch.tensor(sample.class_id, dtype=torch.long)

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
            "class_name": self.class_names[sample.class_id],
        }
        return image, target, meta

    def _scan_samples(self, surfaces: Iterable[SurfaceType]) -> list[SDNET2018Sample]:
        samples: list[SDNET2018Sample] = []

        for surface in surfaces:
            if surface not in self.LABEL_DIRS:
                valid = ", ".join(self.LABEL_DIRS)
                raise ValueError(f"Unknown surface '{surface}'. Valid surfaces: {valid}")

            for label_dir in self.LABEL_DIRS[surface]:
                folder = self.root / surface / label_dir
                if not folder.exists():
                    continue

                cracked = label_dir.startswith("C")
                class_id = self._class_id(surface, label_dir, cracked)
                for image_path in sorted(folder.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in self.image_extensions:
                        samples.append(
                            SDNET2018Sample(
                                image_path=image_path,
                                surface=surface,
                                label_dir=label_dir,
                                cracked=cracked,
                                class_id=class_id,
                            )
                        )

        return samples

    def _class_id(self, surface: SurfaceType, label_dir: str, cracked: bool) -> int:
        if self.class_mode == "binary":
            return 1 if cracked else 0
        return self.class_names.index(label_dir)

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        data = data.view(image.height, image.width, len(image.getbands()))
        return data.permute(2, 0, 1).contiguous().float().div(255.0)


def create_dataset(
    path: str | Path,
    *,
    class_mode: ClassMode = "binary",
    return_meta: bool = True,
    transform: Callable | None = None,
) -> tuple[SDNET2018ClassificationDataset, int]:
    dataset = SDNET2018ClassificationDataset(
        path,
        class_mode=class_mode,
        return_meta=return_meta,
        transform=transform,
    )
    return dataset, dataset.num_classes


def split_dataset(
    dataset: SDNET2018ClassificationDataset,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_count = int(round(len(indices) * train_ratio))
    return Subset(dataset, indices[:train_count]), Subset(dataset, indices[train_count:])


if __name__ == "__main__":
    dataset, class_num = create_dataset("SDNET2018")
    image, target, meta = dataset[0]

    class_counts = {name: 0 for name in dataset.class_names}
    for sample in dataset.samples:
        class_counts[dataset.class_names[sample.class_id]] += 1

    print(f"samples: {len(dataset)}")
    print(f"class_num: {class_num}")
    print(f"class_names: {dataset.class_names}")
    print(f"class_counts: {class_counts}")
    print(f"image shape: {tuple(image.shape)}")
    print(f"target: {target.item()}")
    print(f"meta: {meta}")
