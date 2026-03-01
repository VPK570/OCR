"""
grouping.py — Stages 3 & 4: Bounding Box Post-processing + Line Clustering

Stage 3: Filter tiny/duplicate boxes.
Stage 4: Cluster remaining boxes into text lines, sort L→R within each line.

Writes:
  02_detection_raw/boxes_cleaned.json      — filtered boxes
  04_grouped_lines_visualized/lines.png    — each line cluster in a distinct color
"""

import os
import cv2
import numpy as np
from typing import List, Tuple

from utils import get_logger, save_json, draw_lines


Box = Tuple[int, int, int, int]   # (x1, y1, x2, y2)


class BoxGrouper:
    """
    Stages 3 & 4 of the pipeline.

    Args:
        min_box_area       : Boxes smaller than this (px²) are discarded.
        iou_threshold      : Boxes with IoU above this are considered duplicates.
        y_center_threshold : Max vertical-center distance (px) to merge into same line.
                             Set to 0 to use adaptive threshold (half median box height).
        output_dir         : If set, saves debug outputs here.
    """

    def __init__(
        self,
        min_box_area: int = 100,
        iou_threshold: float = 0.5,
        y_center_threshold: int = 0,   # 0 = adaptive
        output_dir: str = None,
    ):
        self.min_box_area = min_box_area
        self.iou_threshold = iou_threshold
        self.y_center_threshold = y_center_threshold
        self.output_dir = output_dir
        self._log = get_logger("BoxGrouper")

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def run(
        self,
        boxes: List[Box],
        image: np.ndarray,
        image_name: str = "image",
    ) -> List[List[Box]]:
        """
        Full pipeline:  filter → cluster → sort → save.

        Returns:
            line_clusters : List of lines. Each line = list of (x1,y1,x2,y2) sorted L→R.
        """
        self._log.info(f"  Input: {len(boxes)} raw boxes")

        # Stage 3 — filter
        cleaned = self.filter_boxes(boxes)
        self._log.info(f"  After filtering: {len(cleaned)} boxes")

        # Stage 4 — cluster
        line_clusters = self.cluster_into_lines(cleaned)
        self._log.info(f"  Clustered into {len(line_clusters)} lines")

        # ── Save cleaned boxes JSON ──
        if self.output_dir:
            json_path = os.path.join(
                self.output_dir, "02_detection_raw", "boxes_cleaned.json"
            )
            save_json(
                {
                    "image": image_name,
                    "num_boxes": len(cleaned),
                    "boxes": [
                        {"box_id": i, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
                        for i, (x1, y1, x2, y2) in enumerate(cleaned)
                    ],
                },
                json_path,
            )
            self._log.info(f"  Saved cleaned boxes → {json_path}")

            # ── Save line cluster visualization ──
            vis = draw_lines(image, line_clusters)
            vis_path = os.path.join(
                self.output_dir, "04_grouped_lines_visualized", "lines.png"
            )
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
            cv2.imwrite(vis_path, vis)
            self._log.info(f"  Saved line visualization → {vis_path}")

        return line_clusters

    def filter_boxes(self, boxes: List[Box]) -> List[Box]:
        """
        Removes tiny boxes and near-duplicate overlapping boxes.
        Sorts output by vertical then horizontal position.
        """
        # 1. Area filter
        filtered = [
            b for b in boxes
            if (b[2] - b[0]) * (b[3] - b[1]) >= self.min_box_area
        ]
        n_before = len(boxes)
        n_after_area = len(filtered)
        self._log.debug(f"    Area filter: {n_before} → {n_after_area}")

        # 2. NMS-style duplicate removal (sort by area desc, suppress overlaps)
        filtered = sorted(
            filtered,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True
        )
        keep = []
        for box in filtered:
            if not any(self._iou(box, k) > self.iou_threshold for k in keep):
                keep.append(box)
        self._log.debug(f"    Duplicate removal: {n_after_area} → {len(keep)}")

        # Sort top-to-bottom then left-to-right
        keep = sorted(keep, key=lambda b: (b[1], b[0]))
        return keep

    def cluster_into_lines(self, boxes: List[Box]) -> List[List[Box]]:
        """
        Groups boxes into text lines using vertical-center proximity.

        Algorithm:
          - Sort boxes by vertical center.
          - For each box, compare its Y-center to the current line's Y-center mean.
          - If within threshold → same line; else → new line.
          - Within each line, sort boxes left-to-right.
        """
        if not boxes:
            return []

        # Adaptive threshold: half the median box height
        thresh = self.y_center_threshold
        if thresh == 0:
            heights = [(b[3] - b[1]) for b in boxes]
            thresh = int(np.median(heights) * 0.6)
            self._log.info(f"  Adaptive Y-threshold: {thresh}px")

        # Sort by vertical center
        boxes_sorted = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)

        lines: List[List[Box]] = []
        current_line: List[Box] = [boxes_sorted[0]]
        current_y_mean = (boxes_sorted[0][1] + boxes_sorted[0][3]) / 2

        for box in boxes_sorted[1:]:
            box_y_center = (box[1] + box[3]) / 2
            if abs(box_y_center - current_y_mean) <= thresh:
                current_line.append(box)
                # Update running mean of Y-centers in this line
                current_y_mean = np.mean(
                    [(b[1] + b[3]) / 2 for b in current_line]
                )
            else:
                # Sort current line L→R before saving
                lines.append(sorted(current_line, key=lambda b: b[0]))
                current_line = [box]
                current_y_mean = box_y_center

        lines.append(sorted(current_line, key=lambda b: b[0]))

        # Final: sort lines top-to-bottom by their average Y
        lines = sorted(lines, key=lambda line: np.mean([(b[1] + b[3]) / 2 for b in line]))

        for i, line in enumerate(lines):
            self._log.debug(f"    Line {i:02d}: {len(line)} boxes")

        return lines

    # ─────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _iou(a: Box, b: Box) -> float:
        """Intersection over Union for two axis-aligned boxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from utils import load_image, load_json

    json_path  = sys.argv[1] if len(sys.argv) > 1 else "../output/02_detection_raw/boxes.json"
    image_path = sys.argv[2] if len(sys.argv) > 2 else "../sample_images/image.png"

    data  = load_json(json_path)
    boxes = [(b["x1"], b["y1"], b["x2"], b["y2"]) for b in data["boxes"]]

    img = load_image(image_path)
    grouper = BoxGrouper(output_dir="../output")
    lines = grouper.run(boxes, img, image_name=data.get("image", "image"))
    print(f"Line clusters: {len(lines)}")
