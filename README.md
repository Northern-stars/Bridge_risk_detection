# Bridge Risk Detection

本项目用于桥梁/混凝土缺陷检测、分类和分割实验。核心工作是把多个来源不同、标注格式不同的数据集统一整理成可训练的数据接口，然后分别支持：

1. YOLO26 目标检测训练与推理。
2. YOLO-World 开放词表目标检测训练与网页推理。
3. SDNET2018 分类训练与测试。
4. DACL bbox 语义分割训练与评估。

生成的数据集和训练结果默认写入：

```text
datasets/
runs/
```

## 仓库结构

当前代码按功能分为三个主要目录：

```text
app/       # 应用程序、推理脚本、网页服务
train/     # 训练脚本、测试脚本、模型定义、训练配置示例
dataset/   # 数据集解析、格式转换、数据集适配器
```

### 应用程序

```text
app/web_yolo_infer.py          # 统一的 YOLO26 / YOLO-World 网页推理入口
app/web_yolo26_infer.py        # 旧版 YOLO26 网页推理入口
app/web_yolo_world_infer.py    # 旧版 YOLO-World 网页推理入口
app/infer_yolo26_image.py      # 单张图片 YOLO26 推理脚本
```

### 训练脚本与模型

```text
train/train_yolo26.py                  # 通用 YOLO26 检测训练脚本
train/train_yolo_world.py              # YOLO-World 检测训练脚本
train/label_texts.example.json         # YOLO-World 类别文本描述示例

train/sdnet2018_classifier_model.py    # 分类模型定义
train/train_sdnet2018_classifier.py    # 分类训练脚本
train/test_sdnet2018_classifier.py     # 分类测试脚本

train/dacl_segmentation_model.py       # DACL 语义分割模型
train/train_dacl_segmentation.py       # DACL 分割训练与评估脚本
```

### 数据集适配器

```text
dataset/combined_yolo_dataset.py         # 合并多个 YOLO 数据集并统一类别映射
dataset/csb_yolo_dataset.py              # CSB crack pixels -> YOLO bbox
dataset/dacl_yolo_dataset.py             # DACL polygon -> YOLO bbox
dataset/multi_defact_yolo_dataset.py     # 原生 YOLO 格式多缺陷数据集
dataset/deepcrack_yolo_dataset.py        # DeepCrack mask -> YOLO bbox
dataset/sdnet2018_yolo_dataset.py        # SDNET2018 分类数据集适配器
dataset/dacl_segmentation_dataset.py     # DACL polygon -> bbox 填充语义分割 mask
```

### 权重与文档

```text
yolo26*.pt                       # 本地 YOLO26 预训练权重
yolov8s-world.pt                 # 本地 YOLO-World 预训练权重
mangazero_panel_ordering_plan.md # MangaZero panel 排序任务技术方案
README.md                        # 项目说明文档
```

## 数据集

### CSB

适配器：

```text
dataset/csb_yolo_dataset.py
```

期望原始结构：

```text
RawDataset/CSB_dataset/
└── entire images/
    ├── crack_train/
    ├── nocrack_train/
    ├── crack_test/
    └── nocrack_test/
```

每张图片对应 JSON 标注，其中包含 `annotations.crack_pixels`。适配器会把 crack pixels 转换成最小外接框，并写成 YOLO 检测格式。

输出：

```text
datasets/csb_yolo/
├── images/train
├── images/val
├── labels/train
├── labels/val
└── data.yaml
```

类别：

```text
0: crack
```

### DACL

适配器：

```text
dataset/dacl_yolo_dataset.py
```

期望原始结构：

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

DACL 标注为 JSON polygon。检测适配器会把每个 polygon 转换成最小外接 YOLO bbox。

主要类别：

```text
Crack, ACrack, Wetspot, Efflorescence, Rust, Rockpocket, Hollowareas,
Cavity, Spalling, Graffiti, Weathering, Restformwork, ExposedRebars,
Bearing, EJoint, Drainage, PEquipment, JTape, WConccor
```

输出：

```text
datasets/dacl_yolo/
```

### Multi Defact

适配器：

```text
dataset/multi_defact_yolo_dataset.py
```

期望原始结构：

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

该数据集的标签已经是 YOLO 格式：

```text
class_id x_center y_center width height
```

当前类别：

```text
0: Cracks
1: Spalling
2: Honeycomb_Surface
3: Exposed_Rebar
4: Seepage
5: Hole
```

输出：

```text
datasets/multi_defact_yolo/
```

### DeepCrack

适配器：

```text
dataset/deepcrack_yolo_dataset.py
```

期望原始结构：

```text
DownloadDataset/Deepcrack/DeepCrack-master/dataset/DeepCrack/
├── train/
└── test/
```

DeepCrack 使用裂缝 mask。适配器会把 mask 前景区域转换成最小外接框。

类别：

```text
0: crack
```

输出：

```text
datasets/deepcrack_yolo/
```

### SDNET2018

适配器：

```text
dataset/sdnet2018_yolo_dataset.py
```

该文件当前作为纯分类数据集适配器使用。

期望原始结构：

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

支持两种分类模式：

```text
binary: uncracked, crack
surface_damage: CD, UD, CP, UP, CW, UW
```

接口：

```python
create_dataset(path, return_meta=False) -> (dataset, class_num)
split_dataset(dataset, train_ratio=0.8, seed=42) -> (train_dataset, val_dataset)
```

## 合并 YOLO 数据集

合并脚本：

```text
dataset/combined_yolo_dataset.py
```

它会合并以下数据集：

```text
CSB
DACL
Multi Defact
DeepCrack
```

输出：

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

类别会被映射到统一 taxonomy，例如：

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

缺失的原始数据集路径会被跳过并打印提示。

### 类别黑名单

`dataset/combined_yolo_dataset.py` 支持类别黑名单。修改文件顶部的：

```python
BLACKLIST_CLASSES = {"graffiti", "weathering"}
```

黑名单支持 canonical 类别名或类别 id。图片仍会保留，但属于黑名单类别的框不会写入合并后的 YOLO label 文件。

### 直接加载已准备数据

如果合并数据集已经生成过，可以设置：

```python
LOAD_PREPARED_DATASET = True
```

或调用：

```python
create_dataset(load=True)
```

这样会直接使用：

```text
datasets/combined_bridge_defect_yolo/data.yaml
```

不会重新准备数据，也不会删除旧数据。如果文件不存在，会直接报错。

## YOLO26 训练

训练入口：

```text
train/train_yolo26.py
```

默认数据集导入：

```python
from dataset.combined_yolo_dataset import create_dataset
DATASET_PATH = None
```

如果要切换到其他 YOLO 数据集，只需要修改文件顶部 import 和 `DATASET_PATH`，例如：

```python
from dataset.csb_yolo_dataset import create_dataset
DATASET_PATH = Path("RawDataset/CSB_dataset")
```

或：

```python
from dataset.dacl_yolo_dataset import create_dataset
DATASET_PATH = Path("RawDataset/dacl")
```

运行：

```bash
python train/train_yolo26.py
```

主要配置也在文件顶部：

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

`MODEL` 指向本地 `.pt` 权重。Ultralytics 会把它作为初始权重加载，训练结果保存到 `runs/detect/yolo26_train/`，不会覆盖原始 `.pt` 文件。

如果要从已有训练继续：

```python
LOAD = True
```

脚本会加载：

```text
runs/detect/yolo26_train/weights/latest.pth
```

并向 Ultralytics 传入 `resume=True`。如果该文件不存在，会直接报错，不会静默退回到 `MODEL`。

数据集生成过程使用 `tqdm` 显示进度条。

## YOLO-World 训练

训练入口：

```text
train/train_yolo_world.py
```

它与 `train/train_yolo26.py` 保持相同的顶部配置风格。数据集仍然通过修改顶部 import 来切换：

```python
from dataset.combined_yolo_dataset import create_dataset
DATASET_PATH = None
```

YOLO-World 额外需要一个 JSON 文件，把类别映射为文本描述：

```python
LABEL_TEXT_JSON = Path("train") / "label_texts.example.json"
```

示例文件：

```text
train/label_texts.example.json
```

支持类别名映射：

```json
{
  "crack": "visible crack damage on bridge or concrete surface",
  "spalling": "spalling concrete with broken or missing surface material"
}
```

支持类别 id 映射：

```json
{
  "0": "visible crack damage on bridge or concrete surface",
  "1": "alligator crack pattern with connected branching cracks"
}
```

也支持列表格式：

```json
[
  {"id": 0, "text": "visible crack damage on bridge or concrete surface"},
  {"name": "spalling", "text": "spalling concrete with broken or missing surface material"}
]
```

脚本会读取原始 YOLO `data.yaml`，把 `names` 替换为文本描述，并写入：

```text
datasets/yolo_world_data.yaml
```

运行：

```bash
python train/train_yolo_world.py
```

主要配置：

```python
MODEL = "yolov8s-world.pt"
EPOCHS = 50
IMGSZ = 640
BATCH = 16
PROJECT = "runs/detect"
NAME = "yolo_world_train"
```

如果当前安装的 Ultralytics 暴露了 `YOLOWorld`，脚本会优先使用它；否则会回退到 `YOLO` 加载 world 权重。

## YOLO26 单图推理

推理入口：

```text
app/infer_yolo26_image.py
```

修改文件顶部的输入图片路径：

```python
IMAGE_PATH = Path("test.jpg")
```

运行：

```bash
python app/infer_yolo26_image.py
```

脚本会按顺序寻找权重：

```text
F:/opt/homebrew/runs/detect/runs/detect/yolo26_train/weights/latest.pth
F:/opt/homebrew/runs/detect/runs/detect/yolo26_train/weights/best.pt
F:/opt/homebrew/runs/detect/runs/detect/yolo26_train/weights/last.pt
yolo26l.pt
```

画框结果保存到：

```text
runs/infer/yolo26/
```

## 网页服务

推荐使用统一网页入口：

```text
app/web_yolo_infer.py
```

该网页服务在同一个前端中支持两种检测模式：

```text
YOLO26
YOLO-World
```

安装依赖：

```bash
pip install flask
```

启动网页服务：

```bash
python app/web_yolo_infer.py
```

浏览器打开：

```text
http://127.0.0.1:7860
```

页面中可以通过模型下拉框切换 `YOLO26` 和 `YOLO-World`。选择 `YOLO-World` 时，会显示文本描述输入框，支持一行一个描述、逗号/分号分隔描述，也支持 JSON 格式，例如：

```json
[
  "visible crack damage on bridge or concrete surface",
  {"text": "spalling concrete with broken or missing surface material"}
]
```

网页会展示：

```text
原图
检测结果图
安全结论
检测目标数量
类别统计
目标框明细
JSON 输出
```

旧版网页入口仍然保留：

```text
app/web_yolo26_infer.py        # 仅 YOLO26，端口 http://127.0.0.1:7860
app/web_yolo_world_infer.py    # 仅 YOLO-World，端口 http://127.0.0.1:7861
```

## 分类训练与测试

相关文件：

```text
train/sdnet2018_classifier_model.py
train/train_sdnet2018_classifier.py
train/test_sdnet2018_classifier.py
```

模型会根据数据集返回的类别数动态初始化分类头：

```python
dataset, class_num = create_dataset(DATASET_PATH)
model = create_model(class_num)
```

如果要切换分类数据集，修改 `train/train_sdnet2018_classifier.py` 和 `train/test_sdnet2018_classifier.py` 顶部的 import。被导入的数据集模块需要提供：

```python
create_dataset(path, return_meta=False) -> (dataset, class_num)
split_dataset(dataset, train_ratio=..., seed=...) -> (train_dataset, val_dataset)
```

训练：

```bash
python train/train_sdnet2018_classifier.py
```

测试：

```bash
python train/test_sdnet2018_classifier.py
```

checkpoint 保存到：

```text
runs/classify/classifier_train/
├── best.pt
└── last.pt
```

## DACL 分割训练

相关文件：

```text
dataset/dacl_segmentation_dataset.py
train/dacl_segmentation_model.py
train/train_dacl_segmentation.py
```

该任务使用 DACL polygon 标注构造语义分割 mask。每个 polygon 先转成最小外接 bbox，然后 bbox 内所有像素填充对应类别 id。类别 `0` 是背景，DACL 原始类别从 `1` 开始。

模型结构：

```text
dilated ResNet50 backbone
optional FPN neck
transformer mask decoder
boundary-guided segmentation head
```

模型同时预测语义分割 logits 和二值边界图。训练目标由多类 Dice loss、交叉熵和边界 BCE loss 组成：

```python
DICE_WEIGHT = 1.0
CE_WEIGHT = 0.2
BOUNDARY_WEIGHT = 0.5
```

运行：

```bash
python train/train_dacl_segmentation.py
```

主要配置在 `train/train_dacl_segmentation.py` 顶部：

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

训练结束后，脚本会加载 `best.pth`，并在 held-out validation 上输出测试指标：

```text
mean_iou
mean_dice
map50
```

`map50` 按语义 mask 计算：对每个前景类别，如果预测类别 mask 与目标类别 mask 的 IoU 大于等于 `0.5`，则记为命中，最后跨类别求平均。

checkpoint 保存到：

```text
runs/segment/dacl_resnet50_transformer_boundary/
├── best.pth
└── latest.pth
```

## MangaZero Panel 排序方案

项目中还包含一份 MangaZero panel 排序任务技术方案：

```text
mangazero_panel_ordering_plan.md
```

该文档讨论了如何使用 MangaZero 构建“给漫画 panel 排序”的任务，包括页面内阅读顺序排序、连续剧情排序、pairwise ranking、set-to-sequence transformer、评价指标和工程目录建议。

## 依赖

安装核心依赖：

```bash
pip install torch torchvision ultralytics pillow tqdm flask
```

本项目不使用命令行参数解析器。大多数脚本通过修改文件顶部的 import、路径常量和配置常量来切换数据集、模型和训练参数。
