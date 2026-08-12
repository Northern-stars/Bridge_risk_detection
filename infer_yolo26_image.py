from __future__ import annotations

from pathlib import Path
import shutil


IMAGE_PATH = Path("test.jpg")
OUTPUT_DIR = Path("runs/infer/yolo26")

PROJECT = Path("F:/opt/homebrew/runs/detect/runs/detect")
RUN_NAME = "yolo26_train"
WEIGHTS_DIR = PROJECT / RUN_NAME / "weights"
WEIGHT_CANDIDATES = (
    WEIGHTS_DIR / "latest.pth",
    WEIGHTS_DIR / "best.pt",
    WEIGHTS_DIR / "last.pt",
    Path("yolo26l.pt"),
)

IMGSZ = 640
CONF = 0.25
DEVICE = None


def main() -> None:
    weight_path = resolve_weight_path()
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Input image does not exist: {IMAGE_PATH}")

    output_path = run_inference(
        image_path=IMAGE_PATH,
        weight_path=weight_path,
        output_dir=OUTPUT_DIR,
    )
    print(f"weights: {weight_path}")
    print(f"input: {IMAGE_PATH}")
    print(f"output: {output_path}")


def run_inference(image_path: Path, weight_path: Path, output_dir: Path) -> Path:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weight_path))

    predict_kwargs = {
        "source": str(image_path),
        "imgsz": IMGSZ,
        "conf": CONF,
        "save": True,
        "project": str(output_dir),
        "name": "predict",
        "exist_ok": True,
    }
    if DEVICE is not None:
        predict_kwargs["device"] = DEVICE

    results = model.predict(**predict_kwargs)
    result = results[0]
    plotted = Path(result.save_dir) / image_path.name

    final_output = output_dir / f"{image_path.stem}_boxed{image_path.suffix}"
    if plotted.exists():
        shutil.copy2(plotted, final_output)
        return final_output

    rendered = result.plot()
    from PIL import Image

    Image.fromarray(rendered[..., ::-1]).save(final_output)
    return final_output


def resolve_weight_path() -> Path:
    for candidate in WEIGHT_CANDIDATES:
        if candidate.exists():
            return candidate

    candidates = "\n".join(f"- {candidate}" for candidate in WEIGHT_CANDIDATES)
    raise FileNotFoundError(f"No YOLO weight file found. Checked:\n{candidates}")


if __name__ == "__main__":
    main()
