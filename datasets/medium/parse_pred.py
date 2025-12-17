"""Prediction parsing utilities.

`generate_labels_csv.py` should import and call `parse_pred_file(...)`.

Expected prediction JSON format (keys are typical for many detectors):
  - labels: list[int]
  - scores: list[float]
  - bboxes: list[list[float]]  # e.g. [x1, y1, x2, y2] (kept as-is)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


# COCO class names (index == label id)
CLASS_NAMES: List[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


DEFAULT_VEHICLE_LABELS: Set[str] = {
    "car",
    "motorcycle",
    "bus",
    "truck",
    "train",
}


def _safe_class_name(label_id: int, class_names: Sequence[str]) -> str:
    if 0 <= label_id < len(class_names):
        return class_names[label_id]
    return f"label_id={label_id}"


def parse_pred_file(
    pred_path: str,
    *,
    image_name: str,
    thresh: float = 0.6,
    class_names: Sequence[str] = CLASS_NAMES,
    vehicle_labels: Set[str] = DEFAULT_VEHICLE_LABELS,
) -> Dict[str, Any]:
    """Parse a single prediction JSON file into a row dict.

    Returns a dict with keys:
      image_name, num_person, num_vehicles, avg_person_conf, total_boxes, bounding_boxes

    Raises:
      FileNotFoundError: if pred_path does not exist (per your requirement).
      ValueError: if JSON is malformed.
    """

    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Missing prediction file for '{image_name}': {pred_path}")

    with open(pred_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    labels = d.get("labels", [])
    scores = d.get("scores", [])
    bboxes = d.get("bboxes", [])

    n = min(len(labels), len(scores), len(bboxes))

    kept: List[Dict[str, Any]] = []
    for i in range(n):
        score = float(scores[i])
        if score < thresh:
            continue

        label_id = int(labels[i])
        name = _safe_class_name(label_id, class_names)

        kept.append({
            "label": name,
            "score": score,
            "bbox": bboxes[i],
        })

    num_person = sum(1 for o in kept if o["label"] == "person")
    num_vehicles = sum(1 for o in kept if o["label"] in vehicle_labels)
    total_boxes = len(kept)

    person_scores = [o["score"] for o in kept if o["label"] == "person"]
    avg_person_conf = (sum(person_scores) / len(person_scores)) if person_scores else 0.0

    return {
        "image_name": image_name,
        "num_person": num_person,
        "num_vehicles": num_vehicles,
        "avg_person_conf": round(avg_person_conf, 3),
        "total_boxes": total_boxes,
        "bounding_boxes": kept,
    }


if __name__ == "__main__":
    # Optional: quick manual test
    import argparse
    import pprint

    ap = argparse.ArgumentParser(description="Parse a single prediction JSON and print the result.")
    ap.add_argument("pred_path", help="Path to the prediction JSON")
    ap.add_argument("--image-name", default="", help="Image name to place in the output")
    ap.add_argument("--thresh", type=float, default=0.6, help="Score threshold")
    args = ap.parse_args()

    img_name = args.image_name or os.path.splitext(os.path.basename(args.pred_path))[0]
    pprint.pprint(parse_pred_file(args.pred_path, image_name=img_name, thresh=args.thresh))
