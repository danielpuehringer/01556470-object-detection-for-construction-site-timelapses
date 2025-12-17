# Idea: Generate a labels CSV by iterating images and parsing per-image prediction files.
'''
Workflow (your requested separation of concerns):
  - This script: iteration over images + CSV writing.
  - parse_pred.py: parsing of each individual prediction JSON.

CSV columns:
  ["image_name", "interesting/not interesting", "num_person", "num_vehicles",
   "avg_person_conf", "total_boxes", "bounding_boxes"]

The field "interesting/not interesting" is set to 0 by default. This is edited manually.
Missing prediction files raise an error.
'''

from __future__ import annotations

import csv
import json
import os

# Import the parsing function from parse_pred.py --> splitting logic into multiple files
from parse_pred import parse_pred_file


IMAGES_DIR = "original"
PREDS_DIR = "preds"
OUTPUT_CSV = "labels.csv"

# Keep consistent with parse_pred defaults
THRESH = 0.6

def iter_image_files(images_dir: str) -> list[str]:
    #Return a sorted list of filenames (not full paths) in images_dir.
    filenames = [
        f for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f))
    ]
    filenames.sort()
    return filenames


def main(
    images_dir: str = IMAGES_DIR,
    preds_dir: str = PREDS_DIR,
    output_csv: str = OUTPUT_CSV,
    thresh: float = THRESH,
) -> None:
    if not os.path.isdir(images_dir):
        raise NotADirectoryError(f"Images directory not found: {images_dir}")
    if not os.path.isdir(preds_dir):
        raise NotADirectoryError(f"Predictions directory not found: {preds_dir}")

    filenames = iter_image_files(images_dir)

    fieldnames = [
        "image_name",
        "interesting/not interesting",
        "num_person",
        "num_vehicles",
        "avg_person_conf",
        "total_boxes",
        "bounding_boxes",
    ]

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filename in filenames:
            # JSON is named in the same way as the image, but with .json extension
            image_stem = os.path.splitext(filename)[0]
            pred_path = os.path.join(preds_dir, f"{image_stem}.json")

            parsed = parse_pred_file(pred_path, image_name=image_stem, thresh=thresh)

            row = {
                "image_name": parsed["image_name"],
                "interesting/not interesting": 0,
                "num_person": parsed["num_person"],
                "num_vehicles": parsed["num_vehicles"],
                "avg_person_conf": parsed["avg_person_conf"],
                "total_boxes": parsed["total_boxes"],
                # Store list-of-dicts as a JSON string in the CSV cell.
                "bounding_boxes": json.dumps(parsed["bounding_boxes"], ensure_ascii=False),
            }
            writer.writerow(row)

    print(f"{output_csv} created with {len(filenames)} entries.")


if __name__ == "__main__":
    main()
