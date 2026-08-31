from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.dacl_segmentation_dataset import create_train_val_datasets
from train.dacl_segmentation_model import create_model


DATASET_PATH = Path("RawDataset/dacl")
RUN_NAME = "dacl_resnet50_transformer_boundary"
OUTPUT_DIR = Path("runs/segment") / RUN_NAME

IMAGE_SIZE = (512,512)
EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
DICE_WEIGHT = 1.0
CE_WEIGHT = 0.2
BOUNDARY_WEIGHT = 0.5
NUM_WORKERS = 4
PRETRAINED_BACKBONE = False
USE_FPN = True
DECODER_CHANNELS = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_QUERIES = 64
LOAD = True
LOAD_CHECKPOINT = OUTPUT_DIR / "latest.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def main() -> None:
    torch.manual_seed(SEED)

    train_dataset, val_dataset, num_classes = create_train_val_datasets(
        DATASET_PATH,
        image_size=IMAGE_SIZE,
        return_meta=False,
    )
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

    model = create_model(
        num_classes,
        pretrained=PRETRAINED_BACKBONE,
        decoder_channels=DECODER_CHANNELS,
        num_heads=TRANSFORMER_HEADS,
        num_queries=TRANSFORMER_QUERIES,
        use_fpn=USE_FPN,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    start_epoch = 1
    best_dice = 0.0
    if LOAD:
        checkpoint = torch.load(LOAD_CHECKPOINT, map_location=DEVICE)
        if int(checkpoint["num_classes"]) != num_classes:
            raise ValueError(f"Checkpoint num_classes={checkpoint['num_classes']}, dataset num_classes={num_classes}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint.get("best_dice", 0.0))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"train samples: {len(train_dataset)}")
    print(f"val/test samples: {len(val_dataset)}")
    print(f"classes: {num_classes} {train_dataset.class_names}")
    print(f"device: {DEVICE}")

    for epoch in range(start_epoch, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, num_classes)
        val_metrics = evaluate(model, val_loader, num_classes, desc="Validation")
        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "num_classes": num_classes,
            "class_names": train_dataset.class_names,
            "dataset_path": str(DATASET_PATH),
            "image_size": IMAGE_SIZE,
            "best_dice": best_dice,
            "val_dice": val_metrics["mean_dice"],
            "decoder_channels": DECODER_CHANNELS,
            "transformer_heads": TRANSFORMER_HEADS,
            "transformer_queries": TRANSFORMER_QUERIES,
            "use_fpn": USE_FPN,
        }

        if val_metrics["mean_dice"] >= best_dice:
            best_dice = val_metrics["mean_dice"]
            checkpoint["best_dice"] = best_dice
            torch.save(checkpoint, OUTPUT_DIR / "best.pth")

        checkpoint["best_dice"] = best_dice
        torch.save(checkpoint, OUTPUT_DIR / "latest.pth")

        print(
            f"epoch {epoch:03d}/{EPOCHS} "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['mean_dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_iou={val_metrics['mean_iou']:.4f} "
            f"val_dice={val_metrics['mean_dice']:.4f} val_map50={val_metrics['map50']:.4f} "
            f"best_dice={best_dice:.4f}"
        )

    best_checkpoint = OUTPUT_DIR / "best.pth"
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])

    test_metrics = evaluate(model, val_loader, num_classes, desc="Test")
    print(
        "test "
        f"mean_iou={test_metrics['mean_iou']:.4f} "
        f"mean_dice={test_metrics['mean_dice']:.4f} "
        f"map50={test_metrics['map50']:.4f}"
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
) -> dict[str, float]:
    model.train()
    metric = SegmentationMetric(num_classes)
    total_loss = 0.0
    total_images = 0

    progress = tqdm(loader, desc="Training", unit="batch")
    for images, masks in progress:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss, loss_parts = segmentation_loss(outputs, masks, num_classes)
        loss.backward()
        optimizer.step()

        predictions = outputs["seg_logits"].argmax(dim=1)
        metric.update(predictions.detach(), masks)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{metric.compute()['mean_dice']:.4f}",
            boundary=f"{loss_parts['boundary']:.4f}",
        )

    metrics = metric.compute()
    metrics["loss"] = total_loss / total_images
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, num_classes: int, *, desc: str) -> dict[str, float]:
    model.eval()
    metric = SegmentationMetric(num_classes)
    total_loss = 0.0
    total_images = 0

    progress = tqdm(loader, desc=desc, unit="batch")
    for images, masks in progress:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        outputs = model(images)
        loss, _ = segmentation_loss(outputs, masks, num_classes)
        predictions = outputs["seg_logits"].argmax(dim=1)
        metric.update(predictions, masks)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size
        current = metric.compute()
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            iou=f"{current['mean_iou']:.4f}",
            dice=f"{current['mean_dice']:.4f}",
        )

    metrics = metric.compute()
    metrics["loss"] = total_loss / total_images
    return metrics


def segmentation_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    seg_logits = outputs["seg_logits"]
    boundary_logits = outputs["boundary_logits"]

    ce_loss = F.cross_entropy(seg_logits, target)
    dice_loss, dice_score = multiclass_dice_loss(seg_logits, target, num_classes)
    boundary_target = mask_to_boundary(target).unsqueeze(1).float()
    boundary_loss = boundary_bce_loss(boundary_logits, boundary_target)

    loss = CE_WEIGHT * ce_loss + DICE_WEIGHT * dice_loss + BOUNDARY_WEIGHT * boundary_loss
    return loss, {
        "ce": ce_loss.item(),
        "dice": dice_loss.item(),
        "dice_score": dice_score.item(),
        "boundary": boundary_loss.item(),
    }


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    *,
    smooth: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = logits.softmax(dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_one_hot, dims)
    cardinality = torch.sum(probs + target_one_hot, dims)
    dice_per_class = (2.0 * intersection + smooth) / (cardinality + smooth)

    dice_score = dice_per_class[1:].mean() if num_classes > 1 else dice_per_class.mean()
    return 1.0 - dice_score, dice_score


def boundary_bce_loss(boundary_logits: torch.Tensor, boundary_target: torch.Tensor) -> torch.Tensor:
    positive = boundary_target.sum()
    negative = boundary_target.numel() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
    return F.binary_cross_entropy_with_logits(boundary_logits, boundary_target, pos_weight=pos_weight)


def mask_to_boundary(mask: torch.Tensor) -> torch.Tensor:
    boundary = torch.zeros_like(mask, dtype=torch.bool)
    boundary[:, :, 1:] |= mask[:, :, 1:] != mask[:, :, :-1]
    boundary[:, :, :-1] |= mask[:, :, 1:] != mask[:, :, :-1]
    boundary[:, 1:, :] |= mask[:, 1:, :] != mask[:, :-1, :]
    boundary[:, :-1, :] |= mask[:, 1:, :] != mask[:, :-1, :]
    return boundary


class SegmentationMetric:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.intersections = torch.zeros(num_classes, dtype=torch.float64)
        self.pred_areas = torch.zeros(num_classes, dtype=torch.float64)
        self.target_areas = torch.zeros(num_classes, dtype=torch.float64)
        self.ap50_hits = torch.zeros(num_classes, dtype=torch.float64)
        self.ap50_targets = torch.zeros(num_classes, dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().cpu()
        target = target.detach().cpu()

        for class_id in range(1, self.num_classes):
            pred_mask = prediction == class_id
            target_mask = target == class_id
            intersection = torch.logical_and(pred_mask, target_mask).sum().item()
            pred_area = pred_mask.sum().item()
            target_area = target_mask.sum().item()

            self.intersections[class_id] += intersection
            self.pred_areas[class_id] += pred_area
            self.target_areas[class_id] += target_area

            for image_index in range(prediction.size(0)):
                image_target = target_mask[image_index]
                if not image_target.any():
                    continue
                self.ap50_targets[class_id] += 1
                image_pred = pred_mask[image_index]
                union = torch.logical_or(image_pred, image_target).sum().item()
                image_intersection = torch.logical_and(image_pred, image_target).sum().item()
                iou = image_intersection / union if union else 0.0
                if iou >= 0.5:
                    self.ap50_hits[class_id] += 1

    def compute(self) -> dict[str, float]:
        union = self.pred_areas + self.target_areas - self.intersections
        valid_iou = self.target_areas > 0
        valid_ap = self.ap50_targets > 0

        iou = torch.zeros_like(self.intersections)
        dice = torch.zeros_like(self.intersections)
        iou[valid_iou] = self.intersections[valid_iou] / union[valid_iou].clamp_min(1.0)
        dice[valid_iou] = (2.0 * self.intersections[valid_iou]) / (
            self.pred_areas[valid_iou] + self.target_areas[valid_iou]
        ).clamp_min(1.0)

        foreground = torch.arange(self.num_classes) > 0
        mean_iou_mask = valid_iou & foreground
        mean_iou = iou[mean_iou_mask].mean().item() if mean_iou_mask.any() else 0.0
        mean_dice = dice[mean_iou_mask].mean().item() if mean_iou_mask.any() else 0.0

        ap50 = torch.zeros_like(self.ap50_hits)
        ap50[valid_ap] = self.ap50_hits[valid_ap] / self.ap50_targets[valid_ap].clamp_min(1.0)
        mean_ap_mask = valid_ap & foreground
        map50 = ap50[mean_ap_mask].mean().item() if mean_ap_mask.any() else 0.0

        return {
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "map50": map50,
        }


if __name__ == "__main__":
    main()
