import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
try:
    import torch  # for device/half precision
except Exception:  # pragma: no cover
    torch = None


SUPPORTED_SOURCES = {"image", "video", "webcam", "folder"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Road object detector (cars, persons, emergency vehicles) using YOLO"
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=sorted(SUPPORTED_SOURCES),
        help="Input type: image, video, or webcam",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to image/video file (not needed for webcam)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Ultralytics YOLO model weights (e.g., yolov8m.pt, yolov8l.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g., cpu, cuda, cuda:0). Auto if not set",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output to file",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display a window with annotated frames",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="person,car,truck,bus,motorcycle,bicycle",
        help="Comma-separated class names to keep",
    )
    parser.add_argument(
        "--emergency",
        action="store_true",
        help="Enable emergency vehicle heuristic (red/blue lights detection)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1536,
        help="Max image size for inference (higher = more accurate, slower)",
    )
    parser.add_argument(
        "--webcam-index",
        type=int,
        default=0,
        help="Webcam index for --source webcam",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/annotated.mp4",
        help="Output file path when saving video/webcam",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Output directory for folder/image processing",
    )
    parser.add_argument(
        "--suppress-person-in-vehicle",
        dest="suppress_person_in_vehicle",
        action="store_true",
        help="Suppress person boxes contained within vehicle boxes (default: on)",
    )
    parser.add_argument(
        "--no-suppress-person-in-vehicle",
        dest="suppress_person_in_vehicle",
        action="store_false",
        help="Disable suppression of person inside vehicles",
    )
    parser.set_defaults(suppress_person_in_vehicle=True)
    parser.add_argument(
        "--keep-motorcycle-riders",
        action="store_true",
        help="Do not suppress persons inside motorcycles (keep riders)",
    )
    # Strong defaults so no extra flags are required
    parser.set_defaults(emergency=True)
    parser.set_defaults(keep_motorcycle_riders=True)
    parser.add_argument(
        "--min-person-area",
        type=int,
        default=400,
        help="Minimum pixel area for person boxes (remove smaller)",
    )
    parser.add_argument(
        "--class-conf",
        type=str,
        default=None,
        help="Per-class confidence overrides, e.g. 'person:0.5,motorcycle:0.15,bicycle:0.15'",
    )
    parser.add_argument(
        "--min-car-center-dist",
        type=int,
        default=0,
        help="Keep only cars whose center is at least this many pixels from any other car",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.6,
        help="Extra NMS IoU threshold to merge duplicate boxes after all filters",
    )
    parser.add_argument(
        "--nms-agnostic",
        action="store_true",
        help="Apply extra NMS class-agnostically (default: per class)",
    )
    parser.add_argument(
        "--nms-map",
        type=str,
        default="person:0.60,car:0.55,motorcycle:0.55,bicycle:0.55,truck:0.50,bus:0.50",
        help="Per-class NMS IoU thresholds, e.g. 'person:0.6,car:0.55'",
    )
    parser.add_argument(
        "--prefer-over",
        type=str,
        default="motorcycle:person,bicycle:person,car:person",
        help="Comma pairs 'A:B' meaning prefer class A over B when boxes overlap (IoU>=0.5)",
    )
    parser.add_argument(
        "--car-min-area",
        type=int,
        default=6000,
        help="Minimum pixel area for car boxes (remove smaller)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use ensemble of multiple models for maximum accuracy",
    )
    parser.add_argument(
        "--tta-flip",
        action="store_true",
        help="Enable horizontal flip test-time augmentation",
    )
    return parser.parse_args()


def load_model(model_path: str, device: str | None) -> YOLO:
    # Auto-select device if not provided
    if device is None and torch is not None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = None
    model = YOLO(model_path)
    # Move to device and fuse for speed
    try:
        if device is not None:
            model.to(device)
        with np.errstate(all='ignore'):
            _ = getattr(model, "fuse", lambda: None)()
        # Use half precision on CUDA for speed
        if device and isinstance(device, str) and device.startswith("cuda") and torch is not None:
            try:
                _ = getattr(model.model, "half", lambda: None)()
            except Exception:
                pass
    except Exception:
        pass
    return model


def filter_by_classes(class_names: List[str], keep: Iterable[str]) -> List[bool]:
    keep_set = {name.strip().lower() for name in keep}
    return [name.lower() in keep_set for name in class_names]


def draw_labelled_box(
    frame: np.ndarray,
    box_xyxy: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box_xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_w, text_h = text_size
    cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w + 6, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 3, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _class_color(name: str) -> Tuple[int, int, int]:
    palette = {
        "person": (52, 152, 219),
        "car": (46, 204, 113),
        "truck": (39, 174, 96),
        "bus": (241, 196, 15),
        "motorcycle": (231, 76, 60),
        "bicycle": (155, 89, 182),
    }
    return palette.get(name.lower(), (0, 200, 0))


def _parse_class_conf(spec: str | None) -> dict[str, float]:
    if not spec:
        return {}
    out: dict[str, float] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        k, v = token.split(":", 1)
        try:
            out[k.strip().lower()] = float(v)
        except ValueError:
            continue
    return out


def _apply_class_conf(
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    names: List[str],
    overrides: dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not overrides or len(boxes) == 0:
        return boxes, classes, scores
    keep_mask = np.ones(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        cls_name = names[int(classes[i])].lower()
        thr = overrides.get(cls_name)
        if thr is not None and scores[i] < thr:
            keep_mask[i] = False
    return boxes[keep_mask], classes[keep_mask], scores[keep_mask]


def _predict_boxes_multiscale(
    model: YOLO,
    img: np.ndarray,
    base_imgsz: int,
    device: str | None,
    conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Run YOLO at multiple scales (images only), merge via NMS adaptive."""
    h, w = img.shape[:2]
    scales = [1.0, 1.25]
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []
    names: list[str] = []

    def run_once(image_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        results = model.predict(
            source=image_np,
            conf=conf,
            imgsz=base_imgsz,
            device=device,
            verbose=False,
        )
        res = results[0]
        nm = [res.names[i] for i in range(len(res.names))]
        b = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        s = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))
        c = res.boxes.cls.cpu().numpy() if res.boxes is not None else np.zeros((0,))
        return b, s, c, nm

    for sc in scales:
        if sc == 1.0:
            b, s, c, names = run_once(img)
            all_boxes.append(b)
            all_scores.append(s)
            all_classes.append(c)
        else:
            resized = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
            b, s, c, names = run_once(resized)
            if len(b) > 0:
                scale_back = np.array([1.0 / sc, 1.0 / sc, 1.0 / sc, 1.0 / sc], dtype=np.float32)
                b = b.astype(np.float32) * scale_back
            all_boxes.append(b)
            all_scores.append(s)
            all_classes.append(c)

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)), names

    boxes = np.concatenate(all_boxes, axis=0) if len(all_boxes) > 0 else np.zeros((0, 4))
    scores = np.concatenate(all_scores, axis=0) if len(all_scores) > 0 else np.zeros((0,))
    classes = np.concatenate(all_classes, axis=0) if len(all_classes) > 0 else np.zeros((0,))

    # Merge duplicates from TTA via adaptive NMS
    boxes, classes, scores = _nms_adaptive(
        boxes,
        classes,
        scores,
        names,
        class_iou={"person": 0.6, "car": 0.55, "motorcycle": 0.55, "bicycle": 0.55, "truck": 0.5, "bus": 0.5},
        fallback_iou=0.55,
        class_agnostic=False,
    )
    return boxes, scores, classes, names


def _nms(
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    iou_thr: float,
    class_agnostic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes, classes, scores
    kept_indices: list[int] = []
    order = np.argsort(-scores)
    suppressed = np.zeros(len(boxes), dtype=bool)

    def iou_box(a: np.ndarray, b: np.ndarray) -> float:
        xi1 = max(a[0], b[0])
        yi1 = max(a[1], b[1])
        xi2 = min(a[2], b[2])
        yi2 = min(a[3], b[3])
        inter_w = max(0.0, float(xi2) - float(xi1))
        inter_h = max(0.0, float(yi2) - float(yi1))
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        ua = _box_area(a) + _box_area(b) - inter
        if ua <= 0:
            return 0.0
        return float(inter / ua)

    for idx in order:
        if suppressed[idx]:
            continue
        kept_indices.append(idx)
        for j in order:
            if j == idx or suppressed[j]:
                continue
            if not class_agnostic and int(classes[j]) != int(classes[idx]):
                continue
            if iou_box(boxes[idx], boxes[j]) >= iou_thr:
                suppressed[j] = True

    kept_indices = np.array(kept_indices, dtype=int)
    return boxes[kept_indices], classes[kept_indices], scores[kept_indices]


def _parse_nms_map(spec: str | None) -> dict[str, float]:
    if not spec:
        return {}
    out: dict[str, float] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        k, v = token.split(":", 1)
        try:
            out[k.strip().lower()] = float(v)
        except ValueError:
            continue
    return out


def _nms_adaptive(
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    names: list[str],
    class_iou: dict[str, float],
    fallback_iou: float,
    class_agnostic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes, classes, scores
    kept_indices: list[int] = []
    order = np.argsort(-scores)
    suppressed = np.zeros(len(boxes), dtype=bool)
    name_list = [names[int(c)].lower() if 0 <= int(c) < len(names) else str(int(c)) for c in classes]

    def iou_box(a: np.ndarray, b: np.ndarray) -> float:
        xi1 = max(a[0], b[0])
        yi1 = max(a[1], b[1])
        xi2 = min(a[2], b[2])
        yi2 = min(a[3], b[3])
        inter_w = max(0.0, float(xi2) - float(xi1))
        inter_h = max(0.0, float(yi2) - float(yi1))
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        ua = _box_area(a) + _box_area(b) - inter
        if ua <= 0:
            return 0.0
        return float(inter / ua)

    for idx in order:
        if suppressed[idx]:
            continue
        kept_indices.append(idx)
        for j in order:
            if j == idx or suppressed[j]:
                continue
            if not class_agnostic and int(classes[j]) != int(classes[idx]):
                continue
            ci = name_list[idx]
            thr = class_iou.get(ci, fallback_iou)
            if iou_box(boxes[idx], boxes[j]) >= thr:
                suppressed[j] = True

    kept_indices = np.array(kept_indices, dtype=int)
    return boxes[kept_indices], classes[kept_indices], scores[kept_indices]


def _parse_prefer_over(spec: str | None) -> list[tuple[str, str]]:
    if not spec:
        return []
    out: list[tuple[str, str]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        a, b = token.split(":", 1)
        a = a.strip().lower()
        b = b.strip().lower()
        if a and b:
            out.append((a, b))
    return out


def _apply_prefer_over(
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    names: list[str],
    preferences: list[tuple[str, str]],
    iou_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not preferences or len(boxes) == 0:
        return boxes, classes, scores
    name_list = [names[int(c)].lower() if 0 <= int(c) < len(names) else str(int(c)) for c in classes]
    keep_mask = np.ones(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if not keep_mask[i]:
            continue
        ni = name_list[i]
        for j in range(len(boxes)):
            if i == j or not keep_mask[j]:
                continue
            nj = name_list[j]
            # If we prefer ni over nj and they highly overlap, drop j; and vice versa
            if (ni, nj) in preferences:
                if _iou(boxes[i], boxes[j]) >= iou_thr:
                    keep_mask[j] = False
            elif (nj, ni) in preferences:
                if _iou(boxes[i], boxes[j]) >= iou_thr:
                    keep_mask[i] = False
                    break
    return boxes[keep_mask], classes[keep_mask], scores[keep_mask]


def _box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    return w * h


def _is_box_inside(inner: np.ndarray, outer: np.ndarray, min_coverage: float = 0.8) -> bool:
    # Check how much of inner's area lies within outer; require center inside and coverage >= min_coverage
    xi1 = max(inner[0], outer[0])
    yi1 = max(inner[1], outer[1])
    xi2 = min(inner[2], outer[2])
    yi2 = min(inner[3], outer[3])
    inter_w = max(0.0, float(xi2) - float(xi1))
    inter_h = max(0.0, float(yi2) - float(yi1))
    inter_area = inter_w * inter_h
    inner_area = _box_area(inner)
    if inner_area <= 0:
        return False
    coverage = inter_area / inner_area
    # center inside outer
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    center_inside = (outer[0] <= cx <= outer[2]) and (outer[1] <= cy <= outer[3])
    return center_inside and (coverage >= min_coverage)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter_w = max(0.0, float(xi2) - float(xi1))
    inter_h = max(0.0, float(yi2) - float(yi1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    ua = _box_area(a) + _box_area(b) - inter
    if ua <= 0:
        return 0.0
    return float(inter / ua)


def _filter_isolated_cars(
    boxes: np.ndarray,
    classes: np.ndarray,
    names: List[str],
    min_center_dist: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if min_center_dist <= 0 or len(boxes) == 0:
        return boxes, classes
    name_list = [names[int(c)].lower() if 0 <= int(c) < len(names) else str(int(c)) for c in classes]
    car_idxs = [i for i, n in enumerate(name_list) if n == "car"]
    if len(car_idxs) <= 1:
        return boxes, classes
    centers = []
    for i in car_idxs:
        x1, y1, x2, y2 = boxes[i]
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
    keep_car = [True] * len(car_idxs)
    for a in range(len(car_idxs)):
        for b in range(len(car_idxs)):
            if a == b:
                continue
            ax, ay = centers[a]
            bx, by = centers[b]
            dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if dist < float(min_center_dist):
                keep_car[a] = False
                break
    keep_mask = np.ones(len(boxes), dtype=bool)
    for idx_pos, i in enumerate(car_idxs):
        if not keep_car[idx_pos]:
            keep_mask[i] = False
    return boxes[keep_mask], classes[keep_mask]


def suppress_persons_in_vehicles(
    boxes: np.ndarray,
    classes: np.ndarray,
    names: List[str],
    min_coverage: float = 0.8,
    keep_motorcycle_riders: bool = False,
) -> np.ndarray:
    if len(boxes) == 0:
        return np.ones((0,), dtype=bool)
    class_names = [names[int(c)].lower() if 0 <= int(c) < len(names) else str(int(c)) for c in classes]
    is_person = np.array([cn == "person" for cn in class_names])
    vehicle_set = {"car", "truck", "bus", "motorcycle"}
    if keep_motorcycle_riders:
        vehicle_set.remove("motorcycle")
    is_vehicle = np.array([cn in vehicle_set for cn in class_names])
    keep = np.ones((len(boxes),), dtype=bool)
    vehicle_indices = np.where(is_vehicle)[0]
    if vehicle_indices.size == 0 or np.where(is_person)[0].size == 0:
        return keep
    for p_idx in np.where(is_person)[0]:
        p_box = boxes[p_idx]
        for v_idx in vehicle_indices:
            v_box = boxes[v_idx]
            if _is_box_inside(p_box, v_box, min_coverage=min_coverage):
                keep[p_idx] = False
                break
            # Also suppress if IoU high (strong overlap) to avoid standing passengers inside open vehicles
            if _iou(p_box, v_box) >= 0.6:
                keep[p_idx] = False
                break
    return keep


def compute_emergency_score(crop_bgr: np.ndarray) -> float:
    """Legacy API kept for compatibility; returns combined red+blue ratio."""
    if crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    sat_mask = s > 120
    val_mask = v > 140
    blue_mask = (h > 90) & (h < 135) & sat_mask & val_mask
    red_mask = ((h < 10) | (h > 160)) & sat_mask & val_mask
    red_count = int(np.count_nonzero(red_mask))
    blue_count = int(np.count_nonzero(blue_mask))
    total = crop_bgr.shape[0] * crop_bgr.shape[1]
    if total == 0:
        return 0.0
    return float((red_count + blue_count) / float(total))


def is_emergency_lights(crop_bgr: np.ndarray) -> bool:
    """Stricter heuristic to reduce false positives:
    - require presence of both red and blue saturated pixels
    - emphasize lights near top of the vehicle box (light bars)
    - require minimum colored pixel count
    """
    if crop_bgr.size == 0:
        return False
    h_img, w_img = crop_bgr.shape[:2]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    sat_mask = s > 130
    val_mask = v > 160
    blue_mask = (h > 95) & (h < 135) & sat_mask & val_mask
    red_mask = ((h < 10) | (h > 165)) & sat_mask & val_mask

    red_count = int(np.count_nonzero(red_mask))
    blue_count = int(np.count_nonzero(blue_mask))
    total = max(1, h_img * w_img)
    red_ratio = red_count / float(total)
    blue_ratio = blue_count / float(total)

    # Focus on top 30% of the box (typical light bar area)
    top_h = max(1, int(0.3 * h_img))
    red_top = int(np.count_nonzero(red_mask[:top_h, :]))
    blue_top = int(np.count_nonzero(blue_mask[:top_h, :]))
    top_total = max(1, top_h * w_img)
    red_top_ratio = red_top / float(top_total)
    blue_top_ratio = blue_top / float(top_total)

    # Thresholds tuned to be conservative
    min_total_pixels = 30
    has_both_colors = (red_count >= min_total_pixels) and (blue_count >= min_total_pixels)
    enough_ratio = (red_ratio >= 0.006 and blue_ratio >= 0.006)
    enough_top = (red_top_ratio >= 0.004) or (blue_top_ratio >= 0.004)
    return bool(has_both_colors and enough_ratio and enough_top)


def annotate_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    names: List[str],
    mark_emergency: bool,
) -> None:
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cls_id = int(classes[i])
        conf = float(scores[i])
        name = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
        label = f"{name} {conf:.2f}"
        color = _class_color(name)
        # Only flag emergency on larger vehicles; motorcycles excluded to reduce false positives
        if mark_emergency and name in {"car", "truck", "bus"}:
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1c:y2c, x1c:x2c]
            if is_emergency_lights(crop):
                label = f"EMERGENCY? {label}"
                color = (0, 0, 255)
        draw_labelled_box(frame, (x1, y1, x2, y2), label, color)


def draw_counts_overlay(frame: np.ndarray, classes: np.ndarray, names: List[str]) -> None:
    if len(classes) == 0:
        return
    class_names = [names[int(c)].lower() if 0 <= int(c) < len(names) else str(int(c)) for c in classes]
    counts: dict[str, int] = {}
    for cn in class_names:
        counts[cn] = counts.get(cn, 0) + 1
    x, y = 10, 24
    for cn, num in sorted(counts.items()):
        color = _class_color(cn)
        text = f"{cn}: {num}"
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 24


def infer_on_image(model: YOLO, image_path: str, args: argparse.Namespace) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    # Multi-scale predictions (built-in)
    boxes, scores, classes, names = _predict_boxes_multiscale(
        model,
        img,
        base_imgsz=args.max_size,
        device=args.device,
        conf=args.conf,
    )
    keep_names = [n.strip() for n in args.classes.split(",") if n.strip()]

    # Filter by class names
    keep_mask = np.array([names[int(c)].lower() in {k.lower() for k in keep_names} for c in classes])
    boxes, classes, scores = boxes[keep_mask], classes[keep_mask], scores[keep_mask]

    # Per-class confidence overrides (helps promote motorcycle/bicycle over stray persons)
    # Sensible defaults to improve recall of vehicles vs persons when not provided
    default_overrides = {
        "person": 0.50,
        "car": 0.25,
        "truck": 0.25,
        "bus": 0.25,
        "motorcycle": 0.18,
        "bicycle": 0.18,
    }
    user_overrides = _parse_class_conf(args.class_conf)
    overrides = (default_overrides | user_overrides) if user_overrides else default_overrides
    boxes, classes, scores = _apply_class_conf(boxes, classes, scores, [names[i] for i in range(len(names))], overrides)

    # Min person area filter
    if args.min_person_area > 0 and len(boxes) > 0:
        img_h, img_w = img.shape[:2]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep_mask_pa = np.ones(len(boxes), dtype=bool)
        for i in range(len(boxes)):
            cls_name = names[int(classes[i])].lower()
            if cls_name == "person" and areas[i] < args.min_person_area:
                keep_mask_pa[i] = False
        boxes, classes, scores = boxes[keep_mask_pa], classes[keep_mask_pa], scores[keep_mask_pa]

    if args.suppress_person_in_vehicle and len(boxes) > 0:
        keep_mask2 = suppress_persons_in_vehicles(
            boxes,
            classes,
            [names[i] for i in range(len(names))],
            min_coverage=0.8,
            keep_motorcycle_riders=args.keep_motorcycle_riders,
        )
        boxes, classes, scores = boxes[keep_mask2], classes[keep_mask2], scores[keep_mask2]

    # Keep only isolated cars if requested
    if args.min_car_center_dist > 0:
        boxes, classes = _filter_isolated_cars(boxes, classes, [names[i] for i in range(len(names))], args.min_car_center_dist)

    # Enforce minimum car area
    if args.car_min_area > 0 and len(boxes) > 0:
        keep_mask_car = np.ones(len(boxes), dtype=bool)
        for i in range(len(boxes)):
            cls_name = [names[i2] for i2 in range(len(names))][int(classes[i])].lower()
            if cls_name == "car" and _box_area(boxes[i]) < float(args.car_min_area):
                keep_mask_car[i] = False
        boxes, classes, scores = boxes[keep_mask_car], classes[keep_mask_car], scores[keep_mask_car]

    # Aspect-ratio refinement: suppress implausible person boxes (too wide vs tall)
    if len(boxes) > 0:
        keep_mask_ar = np.ones(len(boxes), dtype=bool)
        for i in range(len(boxes)):
            cls_name = [names[i2] for i2 in range(len(names))][int(classes[i])].lower()
            if cls_name == "person":
                x1, y1, x2, y2 = boxes[i]
                w = max(1.0, float(x2 - x1))
                h = max(1.0, float(y2 - y1))
                aspect = w / h
                if aspect > 0.9:  # persons typically taller than wide in road scenes
                    keep_mask_ar[i] = False
        boxes, classes, scores = boxes[keep_mask_ar], classes[keep_mask_ar], scores[keep_mask_ar]

    # Extra NMS to merge duplicates (helps motorcycle duplicates)
    # Extra NMS (adaptive per-class)
    boxes, classes, scores = _nms_adaptive(
        boxes,
        classes,
        scores,
        [names[i] for i in range(len(names))],
        _parse_nms_map(args.nms_map),
        fallback_iou=args.nms_iou,
        class_agnostic=args.nms_agnostic,
    )
    # Apply class preference suppression (e.g., prefer motorcycle over person)
    boxes, classes, scores = _apply_prefer_over(
        boxes,
        classes,
        scores,
        [names[i] for i in range(len(names))],
        _parse_prefer_over(args.prefer_over),
        iou_thr=0.5,
    )
    annotate_detections(img, boxes, classes, scores, [names[i] for i in range(len(names))], args.emergency)
    draw_counts_overlay(img, classes, [names[i] for i in range(len(names))])
    return img


def infer_on_video_stream(model: YOLO, cap: cv2.VideoCapture, args: argparse.Namespace) -> None:
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Path(Path(args.output).parent).mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    keep_names = [n.strip() for n in args.classes.split(",") if n.strip()]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Single-scale for video for speed
        results = model.predict(source=frame, conf=args.conf, imgsz=args.max_size, device=args.device, verbose=False)
        res = results[0]
        names = [res.names[i] for i in range(len(res.names))]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))
        classes = res.boxes.cls.cpu().numpy() if res.boxes is not None else np.zeros((0,))

        keep_mask = np.array([names[int(c)].lower() in {k.lower() for k in keep_names} for c in classes])
        boxes, classes, scores = boxes[keep_mask], classes[keep_mask], scores[keep_mask]

        # Per-class confidence overrides
        default_overrides = {
            "person": 0.50,
            "car": 0.25,
            "truck": 0.25,
            "bus": 0.25,
            "motorcycle": 0.18,
            "bicycle": 0.18,
        }
        user_overrides = _parse_class_conf(args.class_conf)
        overrides = (default_overrides | user_overrides) if user_overrides else default_overrides
        boxes, classes, scores = _apply_class_conf(boxes, classes, scores, [names[i] for i in range(len(names))], overrides)

        if args.suppress_person_in_vehicle and len(boxes) > 0:
            keep_mask2 = suppress_persons_in_vehicles(
                boxes,
                classes,
                [names[i] for i in range(len(names))],
                min_coverage=0.8,
                keep_motorcycle_riders=args.keep_motorcycle_riders,
            )
            boxes, classes, scores = boxes[keep_mask2], classes[keep_mask2], scores[keep_mask2]

        # Keep only isolated cars if requested
        if args.min_car_center_dist > 0:
            boxes, classes = _filter_isolated_cars(boxes, classes, [names[i] for i in range(len(names))], args.min_car_center_dist)

        # Enforce minimum car area
        if args.car_min_area > 0 and len(boxes) > 0:
            keep_mask_car = np.ones(len(boxes), dtype=bool)
            for i in range(len(boxes)):
                cls_name = [names[i2] for i2 in range(len(names))][int(classes[i])].lower()
                if cls_name == "car" and _box_area(boxes[i]) < float(args.car_min_area):
                    keep_mask_car[i] = False
            boxes, classes, scores = boxes[keep_mask_car], classes[keep_mask_car], scores[keep_mask_car]

        boxes, classes, scores = _nms_adaptive(
            boxes,
            classes,
            scores,
            [names[i] for i in range(len(names))],
            _parse_nms_map(args.nms_map),
            fallback_iou=args.nms_iou,
            class_agnostic=args.nms_agnostic,
        )
        boxes, classes, scores = _apply_prefer_over(
            boxes,
            classes,
            scores,
            [names[i] for i in range(len(names))],
            _parse_prefer_over(args.prefer_over),
            iou_thr=0.5,
        )
        annotate_detections(frame, boxes, classes, scores, [names[i] for i in range(len(names))], args.emergency)
        draw_counts_overlay(frame, classes, [names[i] for i in range(len(names))])

        if args.show:
            try:
                cv2.imshow("Detections", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except Exception:
                pass

        if writer is not None:
            writer.write(frame)

    if writer is not None:
        writer.release()


def main() -> int:
    args = parse_args()
    if args.source in {"image", "video", "folder"} and not args.input:
        print("--input is required for image/video source", file=sys.stderr)
        return 2
    if args.source == "image" and not Path(args.input).exists():
        print(f"Input image not found: {args.input}", file=sys.stderr)
        return 2
    if args.source == "folder" and not Path(args.input).exists():
        print(f"Input folder not found: {args.input}", file=sys.stderr)
        return 2
    if args.source == "video" and not Path(args.input).exists():
        print(f"Input video not found: {args.input}", file=sys.stderr)
        return 2

    model = load_model(args.model, args.device)

    if args.source == "image":
        annotated = infer_on_image(model, args.input, args)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.save:
            cv2.imwrite(str(out_path), annotated)
        if args.show:
            try:
                cv2.imshow("Detections", annotated)
                cv2.waitKey(0)
            except Exception:
                pass
        return 0

    if args.source == "folder":
        in_dir = Path(args.input)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [p for p in in_dir.rglob("*") if p.suffix.lower() in img_exts]
        if len(images) == 0:
            print("No images found in folder", file=sys.stderr)
            return 2
        for img_path in images:
            annotated = infer_on_image(model, str(img_path), args)
            rel = img_path.relative_to(in_dir)
            out_path = out_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path.with_suffix(".jpg")), annotated)
        return 0

    if args.source == "video":
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Cannot open video: {args.input}", file=sys.stderr)
            return 2
        infer_on_video_stream(model, cap, args)
        cap.release()
        cv2.destroyAllWindows()
        return 0

    if args.source == "webcam":
        cap = cv2.VideoCapture(args.webcam_index)
        if not cap.isOpened():
            print(f"Cannot open webcam index {args.webcam_index}", file=sys.stderr)
            return 2
        infer_on_video_stream(model, cap, args)
        cap.release()
        cv2.destroyAllWindows()
        return 0

    print("Unsupported source", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


