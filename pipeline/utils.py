"""
utils.py — Shared helpers for the prescription digitization pipeline.
Covers: logging, I/O, image drawing, directory management.
"""

import os
import json
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Any, Dict


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Returns a named logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)-8s  %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


# ─────────────────────────────────────────────
# Directory Management
# ─────────────────────────────────────────────

OUTPUT_SUBDIRS = [
    "01_preprocessed",
    "02_detection_raw",
    "03_detection_visualized",
    "04_grouped_lines_visualized",
    "05_line_crops",
    "06_recognition_raw",
    "07_recognition_with_confidence",
    "08_final_text",
    "09_error_analysis",
]

def setup_output_dirs(base_dir: str) -> Dict[str, str]:
    """
    Creates all 9 output subdirectories under base_dir.
    Returns a dict mapping short name → absolute path.
    """
    paths = {}
    for sub in OUTPUT_SUBDIRS:
        full = os.path.join(base_dir, sub)
        os.makedirs(full, exist_ok=True)
        key = sub.split("_", 1)[1]   # e.g. "preprocessed", "detection_raw"
        paths[sub] = full
    return paths


# ─────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """
    Loads an image as a BGR numpy array.
    Raises FileNotFoundError with a clear message if missing.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def save_json(data: Any, path: str) -> None:
    """Saves any JSON-serialisable object to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    """Loads a JSON file from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, path: str) -> None:
    """Saves a plain text string to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ─────────────────────────────────────────────
# Drawing Helpers
# ─────────────────────────────────────────────

def draw_boxes(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    labels: List[str] = None,
) -> np.ndarray:
    """
    Draws axis-aligned bounding boxes on a copy of image.
    boxes: list of (x1, y1, x2, y2)
    labels: optional list of strings to draw above each box
    """
    vis = image.copy()
    if len(vis.shape) == 2:                     # grayscale → BGR for coloured drawing
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if labels and i < len(labels):
            cv2.putText(
                vis, str(labels[i]),
                (x1, max(y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
            )
    return vis


def draw_lines(
    image: np.ndarray,
    line_clusters: List[List[Tuple[int, int, int, int]]],
) -> np.ndarray:
    """
    Draws each line cluster in a distinct color with a line-ID label.
    line_clusters: list of lines, each line is a list of (x1,y1,x2,y2) boxes.
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # Generate visually distinct colors using HSV
    n = max(len(line_clusters), 1)
    colors = [
        tuple(int(c) for c in cv2.cvtColor(
            np.uint8([[[int(179 * i / n), 220, 220]]]),
            cv2.COLOR_HSV2BGR
        )[0][0])
        for i in range(n)
    ]

    for line_id, (line_boxes, color) in enumerate(zip(line_clusters, colors)):
        for (x1, y1, x2, y2) in line_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Draw bounding rect for the entire line + label
        if line_boxes:
            lx1 = min(b[0] for b in line_boxes)
            ly1 = min(b[1] for b in line_boxes)
            lx2 = max(b[2] for b in line_boxes)
            ly2 = max(b[3] for b in line_boxes)
            cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), color, 1)
            cv2.putText(
                vis, f"L{line_id}",
                (lx1, max(ly1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
            )
    return vis


def overlay_text_on_crop(
    crop: np.ndarray,
    text: str,
    confidence: float,
    flagged: bool = False,
) -> np.ndarray:
    """
    Renders recognised text + confidence score above a line crop image.
    Red border if flagged (low confidence).
    """
    h, w = crop.shape[:2]
    banner_h = 30
    canvas = np.ones((h + banner_h, w, 3), dtype=np.uint8) * 255

    # Place crop (convert to BGR if grayscale)
    if len(crop.shape) == 2:
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    else:
        crop_bgr = crop
    canvas[banner_h:, :] = crop_bgr

    # Text banner
    color = (0, 0, 200) if flagged else (30, 30, 30)
    label = f"{text[:60]}  [conf={confidence:.2f}]{'  ⚠ LOW' if flagged else ''}"
    cv2.putText(canvas, label, (4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if flagged:
        cv2.rectangle(canvas, (0, 0), (w - 1, h + banner_h - 1), (0, 0, 200), 3)
    return canvas
