"""Shared pieces of the build scripts: export, calibration images, and the release quality gate.

Every packaged model is compared with the FP32 ONNX export on COCO128 images that were not used
for calibration. A build whose detections drift too far from FP32 fails instead of being published.
"""
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

IMG_SIZE = 320
NUM_CLASSES = 80
ANCHORS = (IMG_SIZE // 8) ** 2 + (IMG_SIZE // 16) ** 2 + (IMG_SIZE // 32) ** 2
RAW_OUTPUT = [1, 4 + NUM_CLASSES, ANCHORS]  # what Frigate yolo-generic parses
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
CALIBRATION_IMAGES = 100  # first 100 COCO128 images calibrate; the remaining 28 are the holdout
SCORE_THRESHOLD = 0.4  # Frigate post-processing threshold
NMS_IOU = 0.4


def model_name():
    name = os.environ.get("MODEL_NAME", "")
    if not name:
        raise RuntimeError("MODEL_NAME is not set")
    return name


def coco128_images(root=Path("datasets")):
    """Return the 128 COCO128 training images, downloading them once."""
    image_dir = root / "coco128" / "images" / "train2017"
    if not image_dir.is_dir() or not any(image_dir.glob("*.jpg")):
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "coco128.zip"
        urllib.request.urlretrieve(COCO128_URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(root)
        archive.unlink()
    images = sorted(image_dir.glob("*.jpg"))
    if len(images) <= CALIBRATION_IMAGES:
        raise RuntimeError(f"COCO128 calibration images are missing: found {len(images)}")
    return images


def split_images(images):
    return images[:CALIBRATION_IMAGES], images[CALIBRATION_IMAGES:]


def load_rgb(path, size=IMG_SIZE):
    """uint8 RGB HWC, stretched to size x size like a Frigate detection region."""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"cannot read image {path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def export_onnx(model, **kwargs):
    """Export the raw head ([1, 84, 2100]); YOLO26's end-to-end [1, 300, 6] head fails here."""
    path = model.export(format="onnx", imgsz=IMG_SIZE, dynamic=False, simplify=True, nms=None, **kwargs)
    assert_raw_onnx(path)
    return path


def assert_raw_onnx(path):
    import onnx

    graph = onnx.load(path).graph
    if len(graph.output) != 1:
        raise RuntimeError(f"expected one ONNX output, got {len(graph.output)}")
    dims = [dim.dim_value for dim in graph.output[0].type.tensor_type.shape.dim]
    if dims != RAW_OUTPUT:
        raise RuntimeError(f"ONNX output is {dims}, want {RAW_OUTPUT}")
    dims = [dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim]
    if dims != [1, 3, IMG_SIZE, IMG_SIZE]:
        raise RuntimeError(f"ONNX input is {dims}, want [1, 3, {IMG_SIZE}, {IMG_SIZE}]")


def onnx_reference(onnx_path, images):
    """FP32 outputs for uint8 RGB HWC images."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return [session.run(None, {name: nchw_float(image)})[0] for image in images]


def nchw_float(image):
    return (image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0).copy()


def _nms(boxes, scores, iou_threshold):
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area[i] + area[rest] - inter
        overlap = np.where(union > 0, inter / np.maximum(union, 1e-9), 0)
        order = rest[overlap <= iou_threshold]
    return keep


def detections(output, threshold=SCORE_THRESHOLD):
    """Frigate yolo-generic post-processing: [(class, score, x1, y1, x2, y2)], at most 20."""
    pred = np.squeeze(np.asarray(output, dtype=np.float32))
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    scores = pred[:, 4:].max(axis=1)
    keep = scores > threshold
    pred, scores = pred[keep], scores[keep]
    if not len(pred):
        return []
    classes = pred[:, 4:].argmax(axis=1)
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return [(int(classes[i]), float(scores[i]), *map(float, boxes[i])) for i in _nms(boxes, scores, NMS_IOU)[:20]]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def agreement(references, outputs, min_score=0.5):
    """Share of FP32 detections (score >= min_score) found again with the same class at IoU > 0.5,
    the extra confident detections (>= min_score) FP32 does not have, and class-score cosine."""
    found = total = extra = 0
    cosines = []
    for ref, out in zip(references, outputs):
        r = np.squeeze(np.asarray(ref, dtype=np.float32))
        o = np.squeeze(np.asarray(out, dtype=np.float32))
        a, b = r[4:].ravel(), o[4:].ravel()
        cosines.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        ref_dets, out_dets = detections(r), detections(o)
        used = set()
        for d in ref_dets:
            match = max(
                (j for j, e in enumerate(out_dets) if e[0] == d[0] and j not in used),
                key=lambda j: iou(d[2:], out_dets[j][2:]),
                default=None,
            )
            hit = match is not None and iou(d[2:], out_dets[match][2:]) > 0.5
            if hit:
                used.add(match)
            if d[1] >= min_score:
                total += 1
                found += hit
        extra += sum(1 for j, e in enumerate(out_dets) if j not in used and e[1] >= min_score)
    return {
        "recall": found / total if total else 1.0,
        "reference_detections": total,
        "extra": extra,
        "class_cosine": float(np.mean(cosines)) if cosines else 0.0,
    }


def gate(label, references, outputs, min_recall, max_extra, min_cosine):
    """Raise unless the packaged model agrees with FP32 on the holdout images."""
    result = agreement(references, outputs)
    line = (
        f"[GATE] {label}: recall {result['recall']:.1%} of {result['reference_detections']} FP32 detections, "
        f"{result['extra']} extra, class cosine {result['class_cosine']:.4f}"
    )
    print(line, flush=True)
    if (result["recall"] < min_recall or result["extra"] > max_extra
            or result["class_cosine"] < min_cosine):
        raise RuntimeError(
            f"{line} fails the gate (recall >= {min_recall:.0%}, extra <= {max_extra}, cosine >= {min_cosine})"
        )
    return result


def write_zip(zip_name, files):
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files:
            path = Path(path)
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"build did not produce a nonempty {path.name}")
            z.write(path, path.name)
