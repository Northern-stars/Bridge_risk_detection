from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train.sdnet2018_classifier_model import create_model
from dataset.sdnet2018_yolo_dataset import create_dataset, split_dataset


# Change this import block to switch classification datasets:
#
# from dataset.sdnet2018_yolo_dataset import create_dataset, split_dataset
# DATASET_PATH = Path("SDNET2018")
#
# The imported dataset module must expose:
# - create_dataset(path, return_meta=False) -> (dataset, class_num)
# - split_dataset(dataset, train_ratio=..., seed=...) -> (train_dataset, test_dataset)
DATASET_PATH = Path("SDNET2018")
RUN_NAME = "classifier_train"
CHECKPOINT_PATH = Path("runs/classify") / RUN_NAME / "best.pt"
BATCH_SIZE = 128
TRAIN_RATIO = 0.8
SEED = 42
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main() -> None:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    class_num = int(checkpoint["class_num"])
    class_names = list(checkpoint.get("class_names", [f"class_{index}" for index in range(class_num)]))

    dataset, dataset_class_num = build_dataset(DATASET_PATH)
    if dataset_class_num != class_num:
        raise ValueError(f"Checkpoint class_num={class_num}, dataset class_num={dataset_class_num}")

    _, test_dataset = split_dataset(dataset, train_ratio=TRAIN_RATIO, seed=SEED)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.startswith("cuda"),
    )

    model = create_model(class_num).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    correct = 0
    total = 0
    per_class_correct = torch.zeros(class_num, dtype=torch.long)
    per_class_total = torch.zeros(class_num, dtype=torch.long)

    for images, targets in test_loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        preds = model(images).argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.numel()

        for class_id in range(class_num):
            mask = targets == class_id
            per_class_total[class_id] += mask.sum().cpu()
            per_class_correct[class_id] += ((preds == targets) & mask).sum().cpu()

    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"samples: {total}")
    print(f"accuracy: {correct / total:.4f}")
    for class_id, class_name in enumerate(class_names):
        class_total = int(per_class_total[class_id])
        class_correct = int(per_class_correct[class_id])
        class_acc = class_correct / class_total if class_total else 0.0
        print(f"{class_id}: {class_name} accuracy={class_acc:.4f} samples={class_total}")


def build_dataset(path: Path):
    try:
        return create_dataset(path, return_meta=False)
    except TypeError:
        return create_dataset(path)


if __name__ == "__main__":
    main()
