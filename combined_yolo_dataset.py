from __future__ import annotations

import os
from pathlib import Path
import shutil

from tqdm import tqdm

from csb_yolo_dataset import create_dataset as create_csb_dataset
from dacl_yolo_dataset import DACL_CLASSES, create_dataset as create_dacl_dataset
from deepcrack_yolo_dataset import create_dataset as create_deepcrack_dataset
from multi_defact_yolo_dataset import MULTI_DEFACT_CLASSES, create_dataset as create_multi_defact_dataset


OUTPUT_ROOT = Path("datasets") / "combined_bridge_defect_yolo"
REBUILD_DATASET = True
LOAD_PREPARED_DATASET = True

CSB_PATH = Path("RawDataset/CSB_dataset")
DACL_PATH = Path("RawDataset/dacl")
MULTI_DEFACT_PATH = Path("RawDataset/multi_defact")
DEEPCRACK_PATH = Path("DownloadDataset/Deepcrack/DeepCrack-master/dataset/DeepCrack")


CANONICAL_CLASS_NAMES = [
    "crack",
    "alligator_crack",
    "wetspot",
    "efflorescence",
    "rust",
    "rockpocket",
    "hollowareas",
    "cavity",
    "spalling",
    "graffiti",
    "weathering",
    "restformwork",
    "exposed_rebars",
    "bearing",
    "expansion_joint",
    "drainage",
    "protective_equipment",
    "joint_tape",
    "washouts_concrete_corrosion",
]
CANONICAL_CLASS_TO_ID = {name: index for index, name in enumerate(CANONICAL_CLASS_NAMES)}

# Classes listed here will be ignored when merging labels. Values can be
# canonical class names such as "graffiti" or canonical class ids such as 9.
BLACKLIST_CLASSES: tuple[str | int, ...] = ("washouts_concrete_corrosion", "efflorescence", "cavity")


DACL_TO_CANONICAL = {
    "Crack": "crack",
    "ACrack": "alligator_crack",
    "Wetspot": "wetspot",
    "Efflorescence": "efflorescence",
    "Rust": "rust",
    "Rockpocket": "rockpocket",
    "Hollowareas": "hollowareas",
    "Cavity": "cavity",
    "Spalling": "spalling",
    "Graffiti": "graffiti",#deletable
    "Weathering": "weathering",
    "Restformwork": "restformwork",
    "ExposedRebars": "exposed_rebars",
    "Bearing": "bearing",
    "EJoint": "expansion_joint",
    "Drainage": "drainage",
    "PEquipment": "protective_equipment",
    "JTape": "joint_tape",
    "WConccor": "washouts_concrete_corrosion",
}


MULTI_DEFACT_TO_CANONICAL = {
    "Cracks": "crack",
    "Spalling": "spalling",
    "Honeycomb_Surface": "rockpocket",
    "Exposed_Rebar": "exposed_rebars",
    "Seepage": "wetspot",
    "Hole": "cavity",
}


DATASET_SPECS = [
    {
        "name": "csb",
        "path": CSB_PATH,
        "create": create_csb_dataset,
        "class_names": ["crack"],
        "class_mapping": {"crack": "crack"},
    },
    {
        "name": "dacl",
        "path": DACL_PATH,
        "create": create_dacl_dataset,
        "class_names": DACL_CLASSES,
        "class_mapping": DACL_TO_CANONICAL,
    },
    {
        "name": "multi_defact",
        "path": MULTI_DEFACT_PATH,
        "create": create_multi_defact_dataset,
        "class_names": MULTI_DEFACT_CLASSES,
        "class_mapping": MULTI_DEFACT_TO_CANONICAL,
    },
    {
        "name": "deepcrack",
        "path": DEEPCRACK_PATH,
        "create": create_deepcrack_dataset,
        "class_names": ["crack"],
        "class_mapping": {"crack": "crack"},
    },
]


def create_dataset(path: str | Path | None = None, *, load: bool | None = None) -> Path:
    """Create a combined Ultralytics YOLO dataset and return data.yaml.

    ``path`` is accepted for interface compatibility with existing training
    scripts but is not used; source paths are configured at the top of this
    file. If ``load`` is true, use the prepared dataset under ``OUTPUT_ROOT``
    directly and skip all source dataset preparation.
    """

    if load is None:
        load = LOAD_PREPARED_DATASET

    data_yaml = OUTPUT_ROOT / "data.yaml"
    if load:
        if not data_yaml.exists():
            raise FileNotFoundError(
                f"Prepared combined dataset not found: {data_yaml}. "
                "Run create_dataset(..., load=False) once to build it first."
            )
        return data_yaml

    if data_yaml.exists() and not REBUILD_DATASET:
        return data_yaml
    if OUTPUT_ROOT.exists() and REBUILD_DATASET:
        shutil.rmtree(OUTPUT_ROOT)

    for spec in DATASET_SPECS:
        source_path = spec["path"]
        if not source_path.exists():
            print(f"Skipping {spec['name']}: missing source path {source_path}")
            continue

        source_yaml = Path(spec["create"](source_path))
        source_root = read_dataset_root(source_yaml)
        id_mapping = build_id_mapping(spec["class_names"], spec["class_mapping"])
        blacklist_ids = build_blacklist_ids(BLACKLIST_CLASSES)
        merge_prepared_dataset(
            dataset_name=spec["name"],
            source_root=source_root,
            id_mapping=id_mapping,
            blacklist_ids=blacklist_ids,
            output_root=OUTPUT_ROOT,
        )

    write_data_yaml(data_yaml)
    return data_yaml


def build_id_mapping(class_names: list[str], class_mapping: dict[str, str]) -> dict[int, int]:
    id_mapping: dict[int, int] = {}
    for source_id, source_name in enumerate(class_names):
        canonical_name = class_mapping.get(source_name, source_name)
        if canonical_name not in CANONICAL_CLASS_TO_ID:
            raise KeyError(f"No canonical class id for source class {source_name!r} -> {canonical_name!r}")
        id_mapping[source_id] = CANONICAL_CLASS_TO_ID[canonical_name]
    return id_mapping


def build_blacklist_ids(blacklist_classes: set[str | int]) -> set[int]:
    blacklist_ids: set[int] = set()
    for class_item in blacklist_classes:
        if isinstance(class_item, int):
            if class_item < 0 or class_item >= len(CANONICAL_CLASS_NAMES):
                raise ValueError(f"Blacklist class id out of range: {class_item}")
            blacklist_ids.add(class_item)
            continue

        if class_item not in CANONICAL_CLASS_TO_ID:
            raise KeyError(f"Unknown blacklist class name: {class_item!r}")
        blacklist_ids.add(CANONICAL_CLASS_TO_ID[class_item])
    return blacklist_ids


def merge_prepared_dataset(
    *,
    dataset_name: str,
    source_root: Path,
    id_mapping: dict[int, int],
    blacklist_ids: set[int],
    output_root: Path,
) -> None:
    for split in ("train", "val", "test"):
        image_dir = source_root / "images" / split
        label_dir = source_root / "labels" / split
        if not image_dir.exists():
            continue

        output_image_dir = output_root / "images" / split
        output_label_dir = output_root / "labels" / split
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        images = [path for path in sorted(image_dir.iterdir()) if path.is_file() and is_image(path)]
        progress = tqdm(images, desc=f"Merging {dataset_name}/{split}", unit="image")
        for image_path in progress:
            output_name = f"{dataset_name}_{image_path.name}"
            output_image = output_image_dir / output_name
            output_label = output_label_dir / f"{Path(output_name).stem}.txt"
            source_label = label_dir / f"{image_path.stem}.txt"

            link_or_copy(image_path, output_image)
            remap_label(source_label, output_label, id_mapping, blacklist_ids)


def remap_label(
    source_label: Path,
    output_label: Path,
    id_mapping: dict[int, int],
    blacklist_ids: set[int],
) -> None:
    output_label.parent.mkdir(parents=True, exist_ok=True)
    if not source_label.exists():
        output_label.write_text("", encoding="utf-8")
        return

    lines: list[str] = []
    for line_number, line in enumerate(source_label.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label at {source_label}:{line_number}: {line!r}")

        source_class_id = int(float(parts[0]))
        if source_class_id not in id_mapping:
            raise KeyError(f"No class mapping for class id {source_class_id} in {source_label}")

        canonical_class_id = id_mapping[source_class_id]
        if canonical_class_id in blacklist_ids:
            continue

        parts[0] = str(canonical_class_id)
        lines.append(" ".join(parts))

    output_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_dataset_root(data_yaml: Path) -> Path:
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("path:"):
            return Path(line.split(":", 1)[1].strip())
    return data_yaml.parent


def write_data_yaml(data_yaml: Path) -> None:
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"path: {OUTPUT_ROOT.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(CANONICAL_CLASS_NAMES)],
        "",
    ]
    data_yaml.write_text("\n".join(lines), encoding="utf-8")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


if __name__ == "__main__":
    yaml_path = create_dataset()
    print(f"Combined dataset yaml: {yaml_path}")
    print(f"Classes: {len(CANONICAL_CLASS_NAMES)}")
