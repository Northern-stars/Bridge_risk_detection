# Bridge Risk Detection

This project converts several bridge/concrete defect datasets into a common
training interface, then trains YOLO26 detection models or lightweight PyTorch
classification models.

The main idea is:

1. Parse raw datasets with different annotation formats.
2. Convert detection datasets into Ultralytics YOLO format.
3. Keep dataset switching simple by changing the import block at the top of the
   training script.
4. Optionally merge multiple YOLO datasets into one canonical defect taxonomy.

## Project Structure

```text
.
├── train_yolo26.py                  # Generic Ultralytics YOLO26 training script
├── combined_yolo_dataset.py         # Merges multiple YOLO datasets with class mapping
├── csb_yolo_dataset.py              # CSB crack pixels -> YOLO bbox
├── dacl_yolo_dataset.py             # DACL polygons -> YOLO bbox
├── multi_defact_yolo_dataset.py     # Native YOLO-format multi-defect dataset
├── deepcrack_yolo_dataset.py        # DeepCrack masks -> YOLO bbox
├── sdnet2018_yolo_dataset.py        # SDNET2018 classification dataset adapter
├── sdnet2018_classifier_model.py    # Classification model definition
├── train_sdnet2018_classifier.py    # Generic classification training script
├── test_sdnet2018_classifier.py     # Generic classification test script
└── yolo26*.pt                       # Local YOLO26 pretrained weights
```

Generated datasets and training results are written under:

```text
datasets/
runs/
```

## Datasets

### CSB Dataset

Adapter: `csb_yolo_dataset.py`

Expected raw structure:

```text
RawDataset/CSB_dataset/
└── entire images/
    ├── crack_train/
    ├── nocrack_train/
    ├── crack_test/
    └── nocrack_test/
```

Each image has JSON annotation data containing `annotations.crack_pixels`.
The adapter converts crack pixels to one minimum bounding box per image.

YOLO output:

```text
datasets/csb_yolo/
├── images/train
├── images/val
├── labels/train
├── labels/val
└── data.yaml
```

Class:

```text
0: crack
```

### DACL

Adapter: `dacl_yolo_dataset.py`

Expected raw structure:

```text
RawDataset/dacl/
└── train_phase/
    ├── images/
    │   ├── train/
    │   └── validation/
    └── annotations/
        ├── train/
        └── validation/
```

DACL annotations are JSON polygon annotations. The adapter converts every
polygon to its minimum enclosing YOLO bounding box.

Main classes include:

```text
Crack, ACrack, Wetspot, Efflorescence, Rust, Rockpocket, Hollowareas,
Cavity, Spalling, Graffiti, Weathering, Restformwork, ExposedRebars,
Bearing, EJoint, Drainage, PEquipment, JTape, WConccor
```

YOLO output:

```text
datasets/dacl_yolo/
```

### Multi Defact

Adapter: `multi_defact_yolo_dataset.py`

Expected raw structure:

```text
RawDataset/multi_defact/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The labels are already in YOLO format:

```text
class_id x_center y_center width height
```

Current classes:

```text
0: Cracks
1: Spalling
2: Honeycomb_Surface
3: Exposed_Rebar
4: Seepage
5: Hole
```

YOLO output:

```text
datasets/multi_defact_yolo/
```

### DeepCrack

Adapter: `deepcrack_yolo_dataset.py`

Expected raw structure:

```text
DownloadDataset/Deepcrack/DeepCrack-master/dataset/DeepCrack/
├── train/
└── test/
```

DeepCrack uses crack masks. The adapter converts foreground mask pixels to a
minimum bounding box.

Class:

```text
0: crack
```

YOLO output:

```text
datasets/deepcrack_yolo/
```

### SDNET2018

Adapter: `sdnet2018_yolo_dataset.py`

This file is now used as a pure classification dataset adapter.

Expected raw structure:

```text
SDNET2018/
├── D/
│   ├── CD/
│   └── UD/
├── P/
│   ├── CP/
│   └── UP/
└── W/
    ├── CW/
    └── UW/
```

Supported classification modes:

```text
binary: uncracked, crack
surface_damage: CD, UD, CP, UP, CW, UW
```

The classification adapter exposes:

```python
create_dataset(path, return_meta=False) -> (dataset, class_num)
split_dataset(dataset, train_ratio=0.8, seed=42) -> (train_dataset, val_dataset)
```

## Combined YOLO Dataset

`combined_yolo_dataset.py` builds a unified YOLO dataset from:

```text
CSB
DACL
Multi Defact
DeepCrack
```

It writes:

```text
datasets/combined_bridge_defect_yolo/
├── images/train
├── images/val
├── images/test
├── labels/train
├── labels/val
├── labels/test
└── data.yaml
```

Class names are normalized into a shared taxonomy. For example:

```text
CSB crack -> crack
DeepCrack crack -> crack
Multi Defact Cracks -> crack
Multi Defact Spalling -> spalling
Multi Defact Honeycomb_Surface -> rockpocket
Multi Defact Exposed_Rebar -> exposed_rebars
Multi Defact Seepage -> wetspot
Multi Defact Hole -> cavity
```

Missing source dataset paths are skipped with a message.

`combined_yolo_dataset.py` also supports a class blacklist. Edit
`BLACKLIST_CLASSES` at the top of the file to stop tracking selected classes:

```python
BLACKLIST_CLASSES = {"graffiti", "weathering"}
```

The blacklist accepts canonical class names or canonical class ids. Images are
still kept in the dataset, but boxes belonging to blacklisted classes are not
written into the merged YOLO label files.

If the combined dataset has already been prepared, set:

```python
LOAD_PREPARED_DATASET = True
```

or call:

```python
create_dataset(load=True)
```

This directly uses `datasets/combined_bridge_defect_yolo/data.yaml` and skips
all source dataset preparation. If the prepared dataset does not exist, it will
raise an error instead of silently rebuilding.

## YOLO26 Training

Training entry:

```text
train_yolo26.py
```

By default it imports:

```python
from combined_yolo_dataset import create_dataset
DATASET_PATH = None
```

To train on another YOLO dataset, edit only the import block at the top of
`train_yolo26.py`, for example:

```python
from csb_yolo_dataset import create_dataset
DATASET_PATH = Path("RawDataset/CSB_dataset")
```

or:

```python
from dacl_yolo_dataset import create_dataset
DATASET_PATH = Path("RawDataset/dacl")
```

Then run:

```bash
python train_yolo26.py
```

Main training settings are also at the top of `train_yolo26.py`:

```python
MODEL = "yolo26n.pt"
LOAD = False
LOAD_WEIGHT_NAME = "latest.pth"
EPOCHS = 100
IMGSZ = 640
BATCH = 32
WORKERS = 12
DEVICE = None
PROJECT = "runs/detect"
NAME = "yolo26_train"
```

`MODEL` points to a local `.pt` file. Ultralytics loads this file as the initial
weights and saves training outputs under `runs/detect/yolo26_train/`; it does
not overwrite the source `.pt` weight file.

To continue training from a saved run checkpoint, set:

```python
LOAD = True
```

The script will load:

```text
runs/detect/yolo26_train/weights/latest.pth
```

and pass `resume=True` to Ultralytics. If that file does not exist, training
will stop with an explicit error instead of falling back to `MODEL`.

Dataset generation uses `tqdm`, so converting images and labels will show a
progress bar before training starts.

## YOLO-World Training

Training entry:

```text
train_yolo_world.py
```

This script follows the same top-level configuration style as `train_yolo26.py`.
The dataset is still selected by changing the import block at the top:

```python
from combined_yolo_dataset import create_dataset
DATASET_PATH = None
```

YOLO-World additionally needs a JSON file that maps class labels to text
descriptions:

```python
LABEL_TEXT_JSON = Path("label_texts.json")
```

An example is provided in:

```text
label_texts.example.json
```

Supported JSON formats include class-name mapping:

```json
{
  "crack": "visible crack damage on bridge or concrete surface",
  "spalling": "spalling concrete with broken or missing surface material"
}
```

class-id mapping:

```json
{
  "0": "visible crack damage on bridge or concrete surface",
  "1": "alligator crack pattern with connected branching cracks"
}
```

or list entries:

```json
[
  {"id": 0, "text": "visible crack damage on bridge or concrete surface"},
  {"name": "spalling", "text": "spalling concrete with broken or missing surface material"}
]
```

The script reads the original YOLO `data.yaml`, replaces the `names` block with
the text descriptions, and writes:

```text
datasets/yolo_world_data.yaml
```

Then run:

```bash
python train_yolo_world.py
```

Main settings:

```python
MODEL = "yolov8s-world.pt"
EPOCHS = 50
IMGSZ = 640
BATCH = 16
PROJECT = "runs/detect"
NAME = "yolo_world_train"
```

If the installed Ultralytics version exposes `YOLOWorld`, the script uses it.
Otherwise it falls back to `YOLO` with the configured world model weights.

## YOLO26 Inference

Single-image inference:

```text
infer_yolo26_image.py
```

Edit the input image path at the top of the file:

```python
IMAGE_PATH = Path("test.jpg")
```

Then run:

```bash
python infer_yolo26_image.py
```

The script searches for weights in this order:

```text
F:/opt/homebrew/runs/detect/runs/detect/yolo26_train/weights/latest.pth
F:/opt/homebrew/runs/detect/runs/detect/yolo26_train/weights/best.pt
F:/opt/homebrew/runs/detect/runs/detect/yolo26_train/weights/last.pt
yolo26l.pt
```

The boxed result is saved under:

```text
runs/infer/yolo26/
```

Simple web UI:

```text
web_yolo26_infer.py
```

Install Flask if needed:

```bash
pip install flask
```

Start the web app:

```bash
python web_yolo26_infer.py
```

Open:

```text
http://127.0.0.1:7860
```

Upload an image and the page will show the original image and the boxed
detection result.

YOLO-World text-prompt web UI:

```text
web_yolo_world_infer.py
```

Start the app:

```bash
python web_yolo_world_infer.py
```

Open:

```text
http://127.0.0.1:7861
```

Upload an image and enter open-vocabulary class descriptions. The text box
accepts one description per line, comma/semicolon separated descriptions, or
JSON formats such as:

```json
[
  "visible crack damage on bridge or concrete surface",
  {"text": "spalling concrete with broken or missing surface material"}
]
```

The page shows parsed text classes, boxed detections, class statistics,
human-readable output analysis, and JSON output.

## Classification Training

Classification files:

```text
sdnet2018_classifier_model.py
train_sdnet2018_classifier.py
test_sdnet2018_classifier.py
```

The model is initialized dynamically from the dataset class count:

```python
dataset, class_num = create_dataset(DATASET_PATH)
model = create_model(class_num)
```

To switch classification datasets, edit the import block at the top of
`train_sdnet2018_classifier.py` and `test_sdnet2018_classifier.py`. The dataset
module must provide:

```python
create_dataset(path, return_meta=False) -> (dataset, class_num)
split_dataset(dataset, train_ratio=..., seed=...) -> (train_dataset, val_dataset)
```

Train:

```bash
python train_sdnet2018_classifier.py
```

Test:

```bash
python test_sdnet2018_classifier.py
```

Checkpoints are saved to:

```text
runs/classify/classifier_train/
├── best.pt
└── last.pt
```

## DACL Segmentation Training

Segmentation files:

```text
dacl_segmentation_dataset.py
dacl_segmentation_model.py
train_dacl_segmentation.py
```

This task uses DACL polygon annotations to create semantic segmentation masks.
Each polygon is converted to its minimum enclosing bbox, and all pixels inside
that bbox are filled with the corresponding class id. Class `0` is background;
DACL classes start from `1`.

The model is a dilated ResNet50 backbone with an optional FPN neck, a
transformer mask decoder, and a boundary-guided segmentation head. It predicts
both semantic segmentation logits and a binary boundary map. The training
objective combines multi-class Dice loss, cross-entropy, and boundary BCE loss:

```python
DICE_WEIGHT = 1.0
CE_WEIGHT = 0.2
BOUNDARY_WEIGHT = 0.5
```

Run:

```bash
python train_dacl_segmentation.py
```

Main settings are at the top of `train_dacl_segmentation.py`:

```python
DATASET_PATH = Path("RawDataset/dacl")
IMAGE_SIZE = (640, 640)
EPOCHS = 50
BATCH_SIZE = 8
PRETRAINED_BACKBONE = False
USE_FPN = True
DECODER_CHANNELS = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_QUERIES = 64
```

After training, the script loads `best.pth` and reports held-out validation
metrics as test metrics:

```text
mean_iou
mean_dice
map50
```

`map50` is computed as class-wise semantic mask AP50: for each foreground class
present in an image, the predicted class mask is counted as a hit when its IoU
with the target class mask is at least `0.5`.

Checkpoints are saved to:

```text
runs/segment/dacl_resnet50_transformer_boundary/
├── best.pth
└── latest.pth
```

## Requirements

Install the core dependencies:

```bash
pip install torch torchvision ultralytics pillow tqdm flask
```

The project assumes the raw datasets are placed under the paths configured at
the top of each dataset/training script. No command-line argument parser is used;
switching datasets is intentionally controlled by editing the top-level imports
and path constants.
