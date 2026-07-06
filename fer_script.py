"""Evaluate every image in Modified_images with FER and save the results to CSV."""

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import fer
try:
    # FER 25.x no longer exports FER from the package root.
    from fer.fer import FER
except ImportError:
    # Compatibility with older FER releases.
    from fer import FER


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CSV_FIELDS = [
    "img_id",
    "version",
    "ambiguous",
    "r_id",
    "directory",
    *EMOTIONS,
    "dom_emotion",
    "response_time_s",
    "timestamp",
]
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_filename(filename):
    """Return (img_id, version, ambiguous) using the human-rating convention."""
    parts = os.path.splitext(os.path.basename(filename))[0].split("_")
    ambiguous = parts[-1] == "a"

    if ambiguous:
        if len(parts) < 3:
            raise ValueError("expected number_emotion_version_a")
        version = parts[-2]
        img_id = "_".join(parts[:-2])
    else:
        if len(parts) < 2:
            raise ValueError("expected number_emotion_version")
        version = parts[-1]
        img_id = "_".join(parts[:-1])

    return img_id, version, ambiguous


def largest_face(detections):
    """Select the largest face when FER finds more than one."""
    return max(
        detections,
        key=lambda detection: detection["box"][2] * detection["box"][3],
    )


def evaluate_images(input_dir, output_csv, use_mtcnn=False):
    input_dir = os.path.normpath(input_dir)
    image_files = sorted(
        filename
        for filename in os.listdir(input_dir)
        if filename.lower().endswith(VALID_EXTENSIONS)
    )

    # OpenCV installations can point cv2.data.haarcascades at a missing or
    # incompatible file. FER ships its own known-good cascade, so use it
    # explicitly instead of relying on OpenCV's global data directory.
    cascade_path = (
        Path(fer.__file__).resolve().parent
        / "data"
        / "haarcascade_frontalface_default.xml"
    )
    if not use_mtcnn:
        cascade_test = cv2.CascadeClassifier(str(cascade_path))
        if not cascade_path.is_file() or cascade_test.empty():
            raise RuntimeError(
                f"FER face cascade could not be loaded: {cascade_path}"
            )

    detector = FER(cascade_file=str(cascade_path), mtcnn=use_mtcnn)
    rows = []

    for index, filename in enumerate(image_files, start=1):
        started = time.perf_counter()
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            detections = []
            print(f"[{index}/{len(image_files)}] Could not read {filename}")
        else:
            detections = detector.detect_emotions(image)

        img_id, version, ambiguous = parse_filename(filename)
        scores = {emotion: "" for emotion in EMOTIONS}
        dominant_emotion = ""

        if detections:
            result = largest_face(detections)
            scores.update(
                {
                    emotion: result["emotions"].get(emotion, "")
                    for emotion in EMOTIONS
                }
            )
            dominant_emotion = max(
                result["emotions"], key=result["emotions"].get
            )
            print(
                f"[{index}/{len(image_files)}] {filename}: "
                f"{dominant_emotion}"
            )
        else:
            print(f"[{index}/{len(image_files)}] {filename}: no face detected")

        rows.append(
            {
                "img_id": img_id,
                "version": version,
                "ambiguous": ambiguous,
                "r_id": "FER",
                "directory": input_dir,
                **scores,
                "dom_emotion": dominant_emotion,
                "response_time_s": round(time.perf_counter() - started, 3),
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            }
        )

    output_parent = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(output_parent, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} results to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a folder of facial images with FER."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="Modified_images",
        help="Image folder (default: Modified_images)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="fer_ratings.csv",
        help="Output CSV path (default: fer_ratings.csv)",
    )
    parser.add_argument(
        "--mtcnn",
        action="store_true",
        help="Use FER's slower MTCNN face detector.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        parser.error(f"image directory does not exist: {args.input_dir}")

    evaluate_images(args.input_dir, args.output, args.mtcnn)


if __name__ == "__main__":
    main()
