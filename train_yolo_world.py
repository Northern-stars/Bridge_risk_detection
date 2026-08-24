from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

# from deepcrack_yolo_dataset import create_dataset
# DATASET_PATH = Path("DownloadDataset/Deepcrack/DeepCrack-master/dataset/DeepCrack")

# from dacl_yolo_dataset import create_dataset
# DATASET_PATH = Path("RawDataset/dacl")

# from multi_defact_yolo_dataset import create_dataset
# DATASET_PATH = Path("RawDataset/multi_defact")

from combined_yolo_dataset import create_dataset
DATASET_PATH = None

MODEL = "yolov8s-world.pt"
LABEL_TEXT_JSON = Path("label_texts.example.json")
GENERATED_DATA_YAML = Path("datasets") / "yolo_world_data.yaml"

LOAD = False
LOAD_WEIGHT_NAME = "latest.pth"
EPOCHS = 50
IMGSZ = 640
BATCH = 16
WORKERS = 12
DEVICE = None
PROJECT = "runs/detect"
NAME = "yolo_world_train"
EXIST_OK = True


def configure_ultralytics_paths(workspace: Path) -> None:
    config_root = workspace / ".ultralytics"
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))


def main() -> None:
    configure_ultralytics_paths(Path.cwd())

    source_data_yaml = Path(create_dataset(DATASET_PATH))
    class_names = read_yolo_names(source_data_yaml)
    label_texts = load_label_texts(LABEL_TEXT_JSON, class_names)
    data_yaml = write_yolo_world_yaml(source_data_yaml, GENERATED_DATA_YAML, label_texts)

    print(f"Using source dataset yaml: {source_data_yaml}")
    print(f"Using YOLO-World dataset yaml: {data_yaml}")
    print(f"Using label text json: {LABEL_TEXT_JSON}")

    model_path = resolve_model_path()
    print(f"Using model: {model_path}")

    model = create_yolo_world_model(model_path)
    set_world_classes_if_supported(model, label_texts)

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "workers": WORKERS,
        "project": PROJECT,
        "name": NAME,
        "exist_ok": EXIST_OK,
    }
    if DEVICE is not None:
        train_kwargs["device"] = DEVICE
    if LOAD:
        train_kwargs["resume"] = True

    model.train(**train_kwargs)


def create_yolo_world_model(model_path: Path | str):
    try:
        from ultralytics import YOLOWorld

        return YOLOWorld(str(model_path))
    except (ImportError, AttributeError):
        from ultralytics import YOLO

        return YOLO(str(model_path))


def set_world_classes_if_supported(model, label_texts: list[str]) -> None:
    set_classes = getattr(model, "set_classes", None)
    if callable(set_classes):
        set_classes(label_texts)


def resolve_model_path() -> Path | str:
    if not LOAD:
        return MODEL

    checkpoint_path = Path(PROJECT) / NAME / "weights" / LOAD_WEIGHT_NAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"LOAD=True but checkpoint does not exist: {checkpoint_path}. "
            "Set LOAD=False to start from MODEL, or put latest.pth in the run weights directory."
        )
    return checkpoint_path


def load_label_texts(json_path: Path, class_names: list[str]) -> list[str]:
    if not json_path.exists():
        raise FileNotFoundError(
            f"Label text json does not exist: {json_path}. "
            "Create it with descriptions for every dataset class."
        )

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    raw = unwrap_label_payload(raw)

    label_texts: list[str | None] = [None] * len(class_names)
    if isinstance(raw, list):
        load_texts_from_list(raw, class_names, label_texts)
    elif isinstance(raw, dict):
        load_texts_from_dict(raw, class_names, label_texts)
    else:
        raise ValueError(f"Unsupported label text json root type: {type(raw).__name__}")

    missing = [class_names[index] for index, text in enumerate(label_texts) if not text]
    if missing:
        raise ValueError(f"Missing text descriptions for classes: {missing}")

    return [str(text) for text in label_texts]


def unwrap_label_payload(raw: Any) -> Any:
    if isinstance(raw, dict):
        for key in ("names", "classes", "labels", "texts", "descriptions"):
            if key in raw:
                return raw[key]
    return raw


def load_texts_from_list(raw: list[Any], class_names: list[str], label_texts: list[str | None]) -> None:
    for index, item in enumerate(raw):
        if isinstance(item, str):
            if index >= len(label_texts):
                raise ValueError(f"Too many label text entries: got index {index}")
            label_texts[index] = item
            continue

        if not isinstance(item, dict):
            raise ValueError(f"Unsupported list item in label text json: {item!r}")

        class_index = item.get("id", item.get("class_id", item.get("index")))
        class_name = item.get("name", item.get("class", item.get("label")))
        text = extract_text_value(item)
        target_index = resolve_label_index(class_index, class_name, class_names)
        label_texts[target_index] = text


def load_texts_from_dict(raw: dict[str, Any], class_names: list[str], label_texts: list[str | None]) -> None:
    class_to_id = {name: index for index, name in enumerate(class_names)}
    for key, value in raw.items():
        if isinstance(value, dict):
            text = extract_text_value(value)
        else:
            text = normalize_text_value(value)

        if is_int_like(key):
            index = int(key)
            if index < 0 or index >= len(label_texts):
                raise ValueError(f"Class id out of range in label text json: {key}")
            label_texts[index] = text
            continue

        if key not in class_to_id:
            raise KeyError(f"Unknown class name in label text json: {key!r}")
        label_texts[class_to_id[key]] = text


def extract_text_value(item: dict[str, Any]) -> str:
    for key in ("text", "description", "prompt", "name", "label"):
        if key in item:
            return normalize_text_value(item[key])
    raise ValueError(f"Label item has no text/description/prompt/name field: {item!r}")


def normalize_text_value(value: Any) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        text = ", ".join(str(part) for part in value)
    else:
        text = str(value)

    text = text.strip()
    if not text:
        raise ValueError("Label text description cannot be empty")
    return text


def resolve_label_index(class_index: Any, class_name: Any, class_names: list[str]) -> int:
    if class_index is not None:
        index = int(class_index)
        if index < 0 or index >= len(class_names):
            raise ValueError(f"Class id out of range in label text json: {class_index}")
        return index

    if class_name is None:
        raise ValueError("Label list item must contain id/class_id/index or name/class/label")

    class_to_id = {name: index for index, name in enumerate(class_names)}
    if str(class_name) not in class_to_id:
        raise KeyError(f"Unknown class name in label text json: {class_name!r}")
    return class_to_id[str(class_name)]


def read_yolo_names(data_yaml: Path) -> list[str]:
    names: dict[int, str] = {}
    in_names = False
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("names:"):
            in_names = True
            continue

        if in_names and re.match(r"^\S", line):
            break

        if in_names:
            match = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if match:
                names[int(match.group(1))] = strip_yaml_scalar(match.group(2))

    if not names:
        raise ValueError(f"No YOLO names block found in {data_yaml}")
    return [names[index] for index in sorted(names)]


def write_yolo_world_yaml(source_yaml: Path, output_yaml: Path, label_texts: list[str]) -> Path:
    output_yaml.parent.mkdir(parents=True, exist_ok=True)

    output_lines: list[str] = []
    in_names = False
    wrote_names = False
    for line in source_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            output_lines.append("names:")
            output_lines.extend(f"  {index}: {quote_yaml_string(text)}" for index, text in enumerate(label_texts))
            in_names = True
            wrote_names = True
            continue

        if in_names:
            if stripped and re.match(r"^\S", line):
                output_lines.append(line)
                in_names = False
            continue

        output_lines.append(line)

    if not wrote_names:
        output_lines.append("names:")
        output_lines.extend(f"  {index}: {quote_yaml_string(text)}" for index, text in enumerate(label_texts))

    output_yaml.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return output_yaml


def quote_yaml_string(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def strip_yaml_scalar(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def is_int_like(value: Any) -> bool:
    try:
        int(str(value))
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    main()
