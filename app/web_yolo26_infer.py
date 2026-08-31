from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
import sys
from uuid import uuid4

from flask import Flask, render_template_string, request, send_from_directory, url_for
from werkzeug.utils import secure_filename


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.infer_yolo26_image import CONF, DEVICE, IMGSZ, resolve_weight_path


APP_ROOT = ROOT_DIR / "runs" / "web_infer" / "yolo26"
UPLOAD_DIR = APP_ROOT / "uploads"
RESULT_DIR = APP_ROOT / "results"
HOST = "127.0.0.1"
PORT = 7860
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


app = Flask(__name__)


PAGE_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bridge Defect Detection</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: #f4f6f8;
      color: #1f2933;
    }
    body {
      margin: 0;
      padding: 32px;
    }
    main {
      max-width: 1120px;
      margin: 0 auto;
    }
    h1 {
      margin: 0 0 20px;
      font-size: 28px;
      font-weight: 700;
    }
    form {
      display: flex;
      gap: 12px;
      align-items: center;
      padding: 16px;
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
    }
    input[type="file"] {
      flex: 1;
      min-width: 0;
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
    }
    .error {
      margin: 16px 0;
      color: #b42318;
      font-weight: 700;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 20px;
    }
    figure {
      margin: 0;
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      overflow: hidden;
    }
    figcaption {
      padding: 12px 14px;
      border-bottom: 1px solid #d9e2ec;
      font-weight: 700;
    }
    img {
      display: block;
      width: 100%;
      height: auto;
    }
    .summary {
      display: grid;
      grid-template-columns: 1.1fr repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 20px;
    }
    .metric,
    .analysis,
    .detail {
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
    }
    .metric {
      padding: 16px;
      min-width: 0;
    }
    .metric-label {
      color: #52606d;
      font-size: 13px;
      margin-bottom: 8px;
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
    .analysis,
    .detail {
      margin-top: 16px;
      overflow: hidden;
    }
    .section-title {
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
    @media (max-width: 760px) {
      body {
        padding: 18px;
      }
      form {
        align-items: stretch;
        flex-direction: column;
      }
      .grid {
        grid-template-columns: 1fr;
      }
      .summary {
        grid-template-columns: 1fr 1fr;
      }
      .metric-value {
        font-size: 20px;
      }
      th,
      td {
        padding: 9px 10px;
      }
    }
    @media (max-width: 520px) {
      .summary {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <h1>Bridge Defect Detection</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <button type="submit">上传并推理</button>
    </form>
    <div class="status">weights: {{ weight_path }}</div>
    {% if error %}
      <div class="error">{{ error }}</div>
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
          <div class="metric-label">缺陷类型</div>
          <div class="metric-value">{{ summary.defect_type_count }} 类</div>
        </div>
        <div class="metric">
          <div class="metric-label">平均置信度</div>
          <div class="metric-value">{{ summary.mean_confidence_label }}</div>
        </div>
      </section>
      <section class="analysis">
        <div class="section-title">简单分析</div>
        <p class="analysis-text">{{ summary.analysis }}</p>
      </section>
      <section class="detail">
        <div class="section-title">缺陷类型统计</div>
        {% if summary.by_class %}
          <table>
            <thead>
              <tr>
                <th>类型</th>
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
          <div class="empty">未检测到桥梁缺陷目标。</div>
        {% endif %}
      </section>
      <section class="detail">
        <div class="section-title">目标识别明细</div>
        {% if summary.detections %}
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>类别</th>
                <th>置信度</th>
                <th>位置 x1,y1,x2,y2</th>
              </tr>
            </thead>
            <tbody>
              {% for item in summary.detections %}
                <tr>
                  <td>{{ loop.index }}</td>
                  <td>{{ item.class_name }}</td>
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

    try:
        weight_path = resolve_weight_path()
    except FileNotFoundError as exc:
        weight_path = "not found"
        error = str(exc)

    if request.method == "POST" and error is None:
        file = request.files.get("image")
        if file is None or file.filename == "":
            error = "请选择一张图片。"
        else:
            try:
                input_path = save_upload(file)
                output_path, summary = infer_uploaded_image(input_path, weight_path)
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


def infer_uploaded_image(image_path: Path, weight_path: Path) -> tuple[Path, dict]:
    from ultralytics import YOLO
    from PIL import Image

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weight_path))
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
    return output_path, summarize_result(result)


def summarize_result(result) -> dict:
    detections = []
    names = result.names
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        classes = boxes.cls.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        xyxy_boxes = boxes.xyxy.detach().cpu().tolist()

        for class_id, confidence, xyxy in zip(classes, confidences, xyxy_boxes):
            class_id = int(class_id)
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
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

    return {
        "is_safe": total_detections == 0,
        "total_detections": total_detections,
        "defect_type_count": len(by_class),
        "mean_confidence": mean_confidence,
        "mean_confidence_label": format_confidence(mean_confidence),
        "max_confidence": max_confidence,
        "max_confidence_label": format_confidence(max_confidence),
        "min_confidence": min_confidence,
        "min_confidence_label": format_confidence(min_confidence),
        "by_class": by_class,
        "detections": detections,
        "analysis": build_analysis(total_detections, by_class, mean_confidence),
    }


def format_confidence(confidence: float | None) -> str:
    if confidence is None:
        return "-"
    return f"{confidence * 100:.1f}%"


def build_analysis(total_detections: int, by_class: list[dict], mean_confidence: float | None) -> str:
    if total_detections == 0:
        return "当前照片未检测到桥梁缺陷目标，判定为安全。"

    defect_parts = [f"{item['class_name']} {item['count']} 处" for item in by_class]
    defect_text = "、".join(defect_parts)
    top_class = by_class[0]
    confidence_text = format_confidence(mean_confidence)
    return (
        f"当前照片检测到 {total_detections} 处疑似缺陷，判定为存在风险；"
        f"缺陷类型包括 {defect_text}。"
        f"平均置信度为 {confidence_text}，其中数量最多的是 {top_class['class_name']} "
        f"({top_class['count']} 处)。请结合右侧检测结果图中的框和类别进行复核。"
    )


if __name__ == "__main__":
    APP_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Open http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
