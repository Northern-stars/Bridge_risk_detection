from __future__ import annotations

import os
from pathlib import Path

# from deepcrack_yolo_dataset import create_dataset
# DATASET_PATH = Path("DownloadDataset/Deepcrack/DeepCrack-master/dataset/DeepCrack")

# from dacl_yolo_dataset import create_dataset
# DATASET_PATH=Path("RawDataset/dacl")

# from multi_defact_yolo_dataset import create_dataset
# DATASET_PATH=("RawDataset/multi_defact")

from csb_yolo_dataset import create_dataset
DATASET_PATH = Path("RawDataset/CSB_dataset")

MODEL = "yolo26n.pt"
EPOCHS = 100
IMGSZ = 640
BATCH = 32
WORKERS = 12
DEVICE = None
PROJECT = "runs/detect"
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
    print(f"Using model: {MODEL}")

    from ultralytics import YOLO

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

    model = YOLO(MODEL)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
