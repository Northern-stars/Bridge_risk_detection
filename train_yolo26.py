from __future__ import annotations

import os
from pathlib import Path

# from deepcrack_yolo_dataset import create_dataset
# DATASET_PATH = Path("DownloadDataset/Deepcrack/DeepCrack-master/dataset/DeepCrack")

# from dacl_yolo_dataset import create_dataset
# DATASET_PATH=Path("RawDataset/dacl")

# from multi_defact_yolo_dataset import create_dataset
# DATASET_PATH=("RawDataset/multi_defact")

from combined_yolo_dataset import create_dataset
DATASET_PATH=None

MODEL = "yolo26l.pt"
LOAD = False
LOAD_WEIGHT_NAME = "latest.pth"
EPOCHS = 50
IMGSZ = 640
BATCH = 16
WORKERS = 12
DEVICE = None
PROJECT = "F:/opt/homebrew/runs/detect/runs/detect"
NAME = "yolo26_train"
EXIST_OK = True


def configure_ultralytics_paths(workspace: Path) -> None:
    config_root = workspace / ".ultralytics"
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))


def main() -> None:
    configure_ultralytics_paths(Path.cwd())

    data_yaml = Path(create_dataset(DATASET_PATH))
    print(f"Using dataset yaml: {data_yaml}")

    from ultralytics import YOLO

    model_path = resolve_model_path()
    print(f"Using model: {model_path}")

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

    model = YOLO(str(model_path))
    model.train(**train_kwargs)


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


if __name__ == "__main__":
    main()
