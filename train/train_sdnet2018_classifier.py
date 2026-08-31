from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train.sdnet2018_classifier_model import create_model
from dataset.sdnet2018_yolo_dataset import create_dataset, split_dataset

DATASET_PATH = Path("SDNET2018")
# Change this import block to switch classification datasets:
#
# from dataset.sdnet2018_yolo_dataset import create_dataset, split_dataset
# DATASET_PATH = Path("SDNET2018")
#
# The imported dataset module must expose:
# - create_dataset(path, return_meta=False) -> (dataset, class_num)
# - split_dataset(dataset, train_ratio=..., seed=...) -> (train_dataset, val_dataset)

RUN_NAME = "classifier_train"
OUTPUT_DIR = Path("runs/classify") / RUN_NAME
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.8
SEED = 42
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    torch.manual_seed(SEED)

    dataset, class_num = build_dataset(DATASET_PATH)
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=TRAIN_RATIO, seed=SEED)
    class_names = get_class_names(dataset, class_num)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.startswith("cuda"),
    )

    model = create_model(class_num).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    print(f"samples: {len(dataset)}")
    print(f"classes: {class_num} {class_names}")
    print(f"device: {DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "class_num": class_num,
            "class_names": class_names,
            "dataset_path": str(DATASET_PATH),
            "val_acc": val_acc,
        }
        torch.save(checkpoint, OUTPUT_DIR / "last.pt")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(checkpoint, OUTPUT_DIR / "best.pt")

        print(
            f"epoch {epoch:03d}/{EPOCHS} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} best_acc={best_acc:.4f}"
        )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def get_class_names(dataset, class_num: int) -> list[str]:
    class_names = getattr(dataset, "class_names", None)
    if class_names is None:
        class_names = getattr(dataset, "classes", None)
    if class_names is None:
        class_names = getattr(dataset, "CLASSES", None)
    if class_names is None:
        return [f"class_{index}" for index in range(class_num)]
    return list(class_names)


def build_dataset(path: Path):
    try:
        return create_dataset(path, return_meta=False)
    except TypeError:
        return create_dataset(path)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return total_loss / total, correct / total


if __name__ == "__main__":
    main()
