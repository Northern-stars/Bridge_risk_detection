from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import re
import shutil
import sys
from uuid import uuid4

from flask import Flask, render_template_string, request, send_from_directory, url_for
from werkzeug.utils import secure_filename


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


APP_ROOT = ROOT_DIR / "runs" / "web_infer"
UPLOAD_DIR = APP_ROOT / "yolo" / "uploads"
RESULT_DIR = APP_ROOT / "results"
HOST = "127.0.0.1"
PORT = 7861
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROJECT = Path("runs/detect")
RUN_NAME = "yolo_world_train"
WEIGHTS_DIR = PROJECT / RUN_NAME / "weights"
WEIGHT_CANDIDATES = (
    WEIGHTS_DIR / "latest.pth",
    WEIGHTS_DIR / "best.pt",
    WEIGHTS_DIR / "last.pt",
)
MODEL = "yolov8s-world.pt"

IMGSZ = 640
CONF = 0.25
DEVICE = None
DEFAULT_TEXT_PROMPTS = """visible crack damage on bridge or concrete surface
spalling concrete with broken or missing surface material
exposed reinforcing steel bar
rust stain or corrosion mark
bridge expansion joint defect"""


app = Flask(__name__)


PAGE_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLO-World Bridge Detection</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: #f4f6f8;
      color: #1f2933;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      padding: 32px;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
    }
    h1 {
      margin: 0 0 20px;
      font-size: 28px;
      font-weight: 700;
    }
    form {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 0.75fr);
      gap: 16px;
      padding: 16px;
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
    }
    label {
      display: block;
      margin-bottom: 8px;
      color: #364152;
      font-size: 14px;
      font-weight: 700;
    }
    input[type="file"],
    textarea {
      width: 100%;
      min-width: 0;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #ffffff;
      color: #1f2933;
      font: inherit;
    }
    input[type="file"] {
      padding: 9px;
    }
    textarea {
      min-height: 168px;
      padding: 10px 12px;
      line-height: 1.55;
      resize: vertical;
    }
    .actions {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      grid-column: 1 / -1;
      gap: 12px;
    }
    button {
      border: 0;
      border-radius: 6px;
      padding: 10px 16px;
      background: #2563eb;
      color: #ffffff;
      font-weight: 700;
      cursor: pointer;
    }
    .status {
      margin: 16px 0;
      color: #52606d;
      overflow-wrap: anywhere;
    }
    .error {
      margin: 16px 0;
      color: #b42318;
      font-weight: 700;
    }
    .summary {
      display: grid;
      grid-template-columns: 1.15fr repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 20px;
    }
    .metric,
    .panel,
    figure {
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
    }
    .metric {
      min-width: 0;
      padding: 16px;
    }
    .metric-label {
      margin-bottom: 8px;
      color: #52606d;
      font-size: 13px;
    }
    .metric-value {
      font-size: 24px;
      font-weight: 700;
      line-height: 1.2;
      overflow-wrap: anywhere;
    }
    .metric.safe .metric-value {
      color: #047857;
    }
    .metric.risky .metric-value {
      color: #b42318;
    }
    .panel {
      margin-top: 16px;
      overflow: hidden;
    }
    .section-title,
    figcaption {
      padding: 12px 14px;
      border-bottom: 1px solid #d9e2ec;
      font-weight: 700;
    }
    .analysis-text {
      margin: 0;
      padding: 14px;
      color: #364152;
      line-height: 1.65;
    }
    .chip-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 14px;
    }
    .chip {
      max-width: 100%;
      padding: 6px 10px;
      border: 1px solid #cbd5e1;
      border-radius: 999px;
      background: #f8fafc;
      color: #364152;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 20px;
    }
    figure {
      margin: 0;
      overflow: hidden;
    }
    img {
      display: block;
      width: 100%;
      height: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th,
    td {
      padding: 10px 14px;
      border-bottom: 1px solid #eef2f6;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }
    th {
      background: #f8fafc;
      color: #364152;
      font-weight: 700;
    }
    tr:last-child td {
      border-bottom: 0;
    }
    .empty {
      padding: 14px;
      color: #52606d;
    }
    pre {
      margin: 0;
      padding: 14px;
      overflow: auto;
      background: #111827;
      color: #e5e7eb;
      font-size: 13px;
      line-height: 1.5;
    }
    @media (max-width: 860px) {
      body {
        padding: 18px;
      }
      form,
      .grid {
        grid-template-columns: 1fr;
      }
      .summary {
        grid-template-columns: 1fr 1fr;
      }
      .metric-value {
        font-size: 20px;
      }
    }
    @media (max-width: 560px) {
      .summary {
        grid-template-columns: 1fr;
      }
      th,
      td {
        padding: 9px 10px;
      }
    }
  </style>
</head>
<body>
  <main>
    <h1>YOLO-World Bridge Detection</h1>
    <form method="post" enctype="multipart/form-data">
      <div>
        <label for="image">图片</label>
        <input id="image" type="file" name="image" accept="image/*" required>
      </div>
      <div>
        <label for="text_prompts">文字描述</label>
        <textarea id="text_prompts" name="text_prompts" required>{{ text_prompts }}</textarea>
      </div>
      <div class="actions">
        <button type="submit">上传并推理</button>
      </div>
    </form>
    <div class="status">weights: {{ weight_path }}</div>
    {% if error %}
      <div class="error">{{ error }}</div>
    {% endif %}
    {% if summary %}
      <section class="panel">
        <div class="section-title">已解析文字类别</div>
        {% if summary.text_classes %}
          <div class="chip-list">
            {% for text_class in summary.text_classes %}
              <span class="chip">{{ loop.index }}. {{ text_class }}</span>
            {% endfor %}
          </div>
        {% else %}
          <div class="empty">未解析到文字类别。</div>
        {% endif %}
      </section>
    {% endif %}
    {% if original_url and result_url %}
      <section class="summary">
        <div class="metric {{ 'safe' if summary.is_safe else 'risky' }}">
          <div class="metric-label">安全结论</div>
          <div class="metric-value">{{ '安全' if summary.is_safe else '存在风险' }}</div>
        </div>
        <div class="metric">
          <div class="metric-label">检测目标</div>
          <div class="metric-value">{{ summary.total_detections }} 处</div>
        </div>
        <div class="metric">
          <div class="metric-label">命中类别</div>
          <div class="metric-value">{{ summary.matched_class_count }} 类</div>
        </div>
        <div class="metric">
          <div class="metric-label">开放词表</div>
          <div class="metric-value">{{ summary.text_class_count }} 类</div>
        </div>
        <div class="metric">
          <div class="metric-label">平均置信度</div>
          <div class="metric-value">{{ summary.mean_confidence_label }}</div>
        </div>
      </section>
      <section class="grid">
        <figure>
          <figcaption>原图</figcaption>
          <img src="{{ original_url }}" alt="Uploaded image">
        </figure>
        <figure>
          <figcaption>检测结果</figcaption>
          <img src="{{ result_url }}" alt="Detection result">
        </figure>
      </section>
      <section class="panel">
        <div class="section-title">输出解析</div>
        <p class="analysis-text">{{ summary.analysis }}</p>
      </section>
      <section class="panel">
        <div class="section-title">类别统计</div>
        {% if summary.by_class %}
          <table>
            <thead>
              <tr>
                <th>文字类别</th>
                <th>数量</th>
                <th>平均置信度</th>
                <th>最高置信度</th>
                <th>最低置信度</th>
              </tr>
            </thead>
            <tbody>
              {% for item in summary.by_class %}
                <tr>
                  <td>{{ item.class_name }}</td>
                  <td>{{ item.count }} 处</td>
                  <td>{{ item.mean_confidence_label }}</td>
                  <td>{{ item.max_confidence_label }}</td>
                  <td>{{ item.min_confidence_label }}</td>
                </tr>
              {% endfor %}
            </tbody>
          </table>
        {% else %}
          <div class="empty">没有命中任何输入的文字类别。</div>
        {% endif %}
      </section>
      <section class="panel">
        <div class="section-title">目标识别明细</div>
        {% if summary.detections %}
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>文字类别</th>
                <th>类别 ID</th>
                <th>置信度</th>
                <th>位置 x1,y1,x2,y2</th>
              </tr>
            </thead>
            <tbody>
              {% for item in summary.detections %}
                <tr>
                  <td>{{ loop.index }}</td>
                  <td>{{ item.class_name }}</td>
                  <td>{{ item.class_id }}</td>
                  <td>{{ item.confidence_label }}</td>
                  <td>{{ item.box_label }}</td>
                </tr>
              {% endfor %}
            </tbody>
          </table>
        {% else %}
          <div class="empty">无目标框。</div>
        {% endif %}
      </section>
      <section class="panel">
        <div class="section-title">JSON 输出</div>
        <pre>{{ summary.json_output }}</pre>
      </section>
    {% endif %}
  </main>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    original_url = None
    result_url = None
    summary = None
    text_prompts = DEFAULT_TEXT_PROMPTS

    try:
        weight_path = resolve_weight_path()
    except FileNotFoundError as exc:
        weight_path = "not found"
        error = str(exc)

    if request.method == "POST":
        text_prompts = request.form.get("text_prompts", "")
        if error is None:
            file = request.files.get("image")
            if file is None or file.filename == "":
                error = "请选择一张图片。"
            else:
                try:
                    text_classes = parse_text_classes(text_prompts)
                    input_path = save_upload(file)
                    output_path, summary = infer_uploaded_image(input_path, weight_path, text_classes)
                    original_url = url_for("serve_upload", filename=input_path.name)
                    result_url = url_for("serve_result", filename=output_path.name)
                except Exception as exc:
                    error = str(exc)

    return render_template_string(
        PAGE_TEMPLATE,
        weight_path=weight_path,
        error=error,
        original_url=original_url,
        result_url=result_url,
        summary=summary,
        text_prompts=text_prompts,
    )


@app.route("/uploads/<path:filename>")
def serve_upload(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/results/<path:filename>")
def serve_result(filename: str):
    return send_from_directory(RESULT_DIR, filename)


def save_upload(file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {suffix}")

    safe_name = secure_filename(file.filename)
    filename = f"{uuid4().hex}_{safe_name}"
    output_path = UPLOAD_DIR / filename
    file.save(output_path)
    return output_path


def infer_uploaded_image(image_path: Path, weight_path: Path | str, text_classes: list[str]) -> tuple[Path, dict]:
    from PIL import Image

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    model = create_yolo_world_model(weight_path)
    set_world_classes(model, text_classes)

    predict_kwargs = {
        "source": str(image_path),
        "imgsz": IMGSZ,
        "conf": CONF,
        "save": True,
        "project": str(RESULT_DIR),
        "name": "_tmp",
        "exist_ok": True,
    }
    if DEVICE is not None:
        predict_kwargs["device"] = DEVICE

    results = model.predict(**predict_kwargs)
    result = results[0]
    plotted = Path(result.save_dir) / image_path.name
    output_path = RESULT_DIR / f"{image_path.stem}_boxed{image_path.suffix}"

    if plotted.exists():
        shutil.copy2(plotted, output_path)
    else:
        rendered = result.plot()
        Image.fromarray(rendered[..., ::-1]).save(output_path)
    return output_path, summarize_result(result, text_classes)


def create_yolo_world_model(model_path: Path | str):
    try:
        from ultralytics import YOLOWorld

        return YOLOWorld(str(model_path))
    except (ImportError, AttributeError):
        from ultralytics import YOLO

        return YOLO(str(model_path))


def set_world_classes(model, text_classes: list[str]) -> None:
    set_classes = getattr(model, "set_classes", None)
    if not callable(set_classes):
        raise AttributeError(
            "当前 Ultralytics 模型不支持 set_classes，无法使用文字描述设置 YOLO-World 开放词表。"
        )
    set_classes(text_classes)


def parse_text_classes(raw_text: str) -> list[str]:
    text = raw_text.strip()
    if not text:
        raise ValueError("请输入至少一个文字描述。")

    if text[0] in "[{":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"文字描述 JSON 解析失败: {exc}") from exc
        classes = parse_json_text_classes(parsed)
    else:
        classes = parse_plain_text_classes(text)

    normalized = []
    seen = set()
    for item in classes:
        value = normalize_text(item)
        key = value.casefold()
        if key not in seen:
            normalized.append(value)
            seen.add(key)

    if not normalized:
        raise ValueError("未解析到有效文字描述。")
    return normalized


def parse_json_text_classes(value) -> list[str]:
    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        classes = []
        for item in value:
            if isinstance(item, dict):
                classes.append(extract_text_value(item))
            else:
                classes.append(item)
        return classes

    if isinstance(value, dict):
        for key in ("names", "classes", "labels", "texts", "descriptions", "prompts"):
            if key in value:
                return parse_json_text_classes(value[key])

        classes = []
        for key, item in value.items():
            if isinstance(item, dict):
                classes.append(extract_text_value(item))
            elif isinstance(item, list):
                classes.append(", ".join(normalize_text(part) for part in item))
            elif isinstance(item, str):
                classes.append(item)
            else:
                classes.append(f"{key}: {item}")
        return classes

    raise ValueError(f"不支持的文字描述 JSON 类型: {type(value).__name__}")


def parse_plain_text_classes(text: str) -> list[str]:
    classes = []
    for line in text.splitlines():
        cleaned = strip_list_marker(line).strip()
        if not cleaned:
            continue
        parts = [part.strip() for part in re.split(r"[;,，；]", cleaned) if part.strip()]
        classes.extend(parts if len(parts) > 1 else [cleaned])
    return classes


def strip_list_marker(text: str) -> str:
    return re.sub(r"^\s*(?:[-*]|\d+[.)、])\s*", "", text)


def extract_text_value(item: dict) -> str:
    for key in ("text", "description", "prompt", "name", "label", "class"):
        if key in item:
            return normalize_text(item[key])
    raise ValueError(f"文字描述条目缺少 text/description/prompt/name 字段: {item!r}")


def normalize_text(value) -> str:
    if isinstance(value, list):
        text = ", ".join(str(part) for part in value)
    else:
        text = str(value)

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("文字描述不能为空。")
    return text


def summarize_result(result, text_classes: list[str]) -> dict:
    detections = parse_detections(result)
    class_groups = defaultdict(list)
    for detection in detections:
        class_groups[detection["class_name"]].append(detection["confidence"])

    by_class = []
    for class_name, confidences in class_groups.items():
        mean_confidence = sum(confidences) / len(confidences)
        by_class.append(
            {
                "class_name": class_name,
                "count": len(confidences),
                "mean_confidence": mean_confidence,
                "mean_confidence_label": format_confidence(mean_confidence),
                "max_confidence": max(confidences),
                "max_confidence_label": format_confidence(max(confidences)),
                "min_confidence": min(confidences),
                "min_confidence_label": format_confidence(min(confidences)),
            }
        )
    by_class.sort(key=lambda item: (-item["count"], -item["mean_confidence"], item["class_name"]))

    total_detections = len(detections)
    mean_confidence = (
        sum(detection["confidence"] for detection in detections) / total_detections
        if total_detections
        else None
    )
    max_confidence = max((detection["confidence"] for detection in detections), default=None)
    min_confidence = min((detection["confidence"] for detection in detections), default=None)

    summary = {
        "is_safe": total_detections == 0,
        "text_classes": text_classes,
        "text_class_count": len(text_classes),
        "total_detections": total_detections,
        "matched_class_count": len(by_class),
        "mean_confidence": mean_confidence,
        "mean_confidence_label": format_confidence(mean_confidence),
        "max_confidence": max_confidence,
        "max_confidence_label": format_confidence(max_confidence),
        "min_confidence": min_confidence,
        "min_confidence_label": format_confidence(min_confidence),
        "by_class": by_class,
        "detections": detections,
    }
    summary["analysis"] = build_analysis(summary)
    summary["json_output"] = json.dumps(build_json_output(summary), ensure_ascii=False, indent=2)
    return summary


def parse_detections(result) -> list[dict]:
    detections = []
    names = result.names
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        classes = boxes.cls.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        xyxy_boxes = boxes.xyxy.detach().cpu().tolist()

        for class_id, confidence, xyxy in zip(classes, confidences, xyxy_boxes):
            class_id = int(class_id)
            class_name = resolve_class_name(names, class_id)
            confidence = float(confidence)
            box = [round(float(value), 1) for value in xyxy]
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "confidence_label": format_confidence(confidence),
                    "box": box,
                    "box_label": ", ".join(f"{value:.1f}" for value in box),
                }
            )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections


def resolve_class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def build_analysis(summary: dict) -> str:
    if summary["total_detections"] == 0:
        return (
            f"已按 {summary['text_class_count']} 个文字描述进行开放词表检测，"
            "当前照片没有命中目标，判定为安全。"
        )

    class_parts = [f"{item['class_name']} {item['count']} 处" for item in summary["by_class"]]
    class_text = "、".join(class_parts)
    top_class = summary["by_class"][0]
    return (
        f"已按 {summary['text_class_count']} 个文字描述进行开放词表检测，"
        f"命中 {summary['matched_class_count']} 类、共 {summary['total_detections']} 处目标，"
        f"判定为存在风险；命中类别包括 {class_text}。"
        f"平均置信度为 {summary['mean_confidence_label']}，"
        f"数量最多的是 {top_class['class_name']} ({top_class['count']} 处)。"
        "请结合检测结果图中的框和类别进行复核。"
    )


def build_json_output(summary: dict) -> dict:
    keys = (
        "is_safe",
        "text_classes",
        "text_class_count",
        "total_detections",
        "matched_class_count",
        "mean_confidence",
        "max_confidence",
        "min_confidence",
        "analysis",
        "by_class",
        "detections",
    )
    return {key: summary[key] for key in keys}


def format_confidence(confidence: float | None) -> str:
    if confidence is None:
        return "-"
    return f"{confidence * 100:.1f}%"


def resolve_weight_path() -> Path | str:
    for candidate in WEIGHT_CANDIDATES:
        if candidate.exists():
            return candidate

    return MODEL


if __name__ == "__main__":
    APP_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Open http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
