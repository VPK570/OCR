"""
crop.py — Stage 5: Line Cropping

For each line cluster output by BoxGrouper:
  - Compute the bounding rectangle covering all word boxes in the line.
  - Crop from the ORIGINAL BGR image (not thresholded — gives best recognition quality).
  - Save each crop as a numbered PNG.

Writes:
  05_line_crops/line_00.png, line_01.png, ...
  05_line_crops/crops_meta.json
"""

import os
import cv2
import numpy as np
from typing import List, Tuple

from utils import get_logger, save_json


Box  = Tuple[int, int, int, int]          # (x1, y1, x2, y2)
Crop = Tuple[int, np.ndarray]              # (line_id, image_crop)


class LineCropper:
    """
    Stage 5 of the pipeline.

    Args:
        padding    : Extra pixels to add around each line's bounding rect.
        output_dir : If set, saves crops and metadata here.
    """

    def __init__(self, padding: int = 8, output_dir: str = None):
        self.padding = padding
        self.output_dir = output_dir
        self._log = get_logger("LineCropper")

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def crop_lines(
        self,
        original_bgr: np.ndarray,
        line_clusters: List[List[Box]],
    ) -> List[Crop]:
        """
        Crops each line cluster from the original BGR image.

        Returns:
            List of (line_id, crop_bgr) tuples, ordered top-to-bottom.
        """
        H, W = original_bgr.shape[:2]
        crops: List[Crop] = []
        meta = []
        pad = self.padding

        for line_id, line_boxes in enumerate(line_clusters):
            if not line_boxes:
                continue

            # Bounding rect of all boxes in this line
            x1 = min(b[0] for b in line_boxes)
            y1 = min(b[1] for b in line_boxes)
            x2 = max(b[2] for b in line_boxes)
            y2 = max(b[3] for b in line_boxes)

            # Apply padding (clamped to image bounds)
            xp1 = max(0, x1 - pad)
            yp1 = max(0, y1 - pad)
            xp2 = min(W, x2 + pad)
            yp2 = min(H, y2 + pad)

            crop = original_bgr[yp1:yp2, xp1:xp2]

            if crop.size == 0:
                self._log.warning(f"  Line {line_id:02d}: empty crop — skipped")
                continue

            self._log.debug(
                f"  Line {line_id:02d}: bbox=({xp1},{yp1},{xp2},{yp2})  "
                f"size={crop.shape[1]}×{crop.shape[0]}"
            )

            crops.append((line_id, crop))
            meta.append({
                "line_id": line_id,
                "x1": xp1, "y1": yp1, "x2": xp2, "y2": yp2,
                "width": xp2 - xp1,
                "height": yp2 - yp1,
                "num_boxes": len(line_boxes),
                "filename": f"line_{line_id:02d}.png",
            })

        self._log.info(f"  Produced {len(crops)} line crops")

        if self.output_dir:
            self._save(crops, meta)

        return crops

    # ─────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────

    def _save(self, crops: List[Crop], meta: list) -> None:
        crops_dir = os.path.join(self.output_dir, "05_line_crops")
        os.makedirs(crops_dir, exist_ok=True)

        for line_id, crop in crops:
            path = os.path.join(crops_dir, f"line_{line_id:02d}.png")
            cv2.imwrite(path, crop)

        meta_path = os.path.join(crops_dir, "crops_meta.json")
        save_json(meta, meta_path)
        self._log.info(f"  Saved crops → {crops_dir}")
        self._log.info(f"  Saved metadata → {meta_path}")


# ─────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from utils import load_image, load_json

    image_path = sys.argv[1] if len(sys.argv) > 1 else "../sample_images/image.png"
    meta_path  = sys.argv[2] if len(sys.argv) > 2 else "../output/02_detection_raw/boxes_cleaned.json"

    data  = load_json(meta_path)
    boxes = [(b["x1"], b["y1"], b["x2"], b["y2"]) for b in data["boxes"]]

    # Fake single-line cluster (all boxes in one line) just for test
    from grouping import BoxGrouper
    img = load_image(image_path)
    grouper = BoxGrouper(output_dir="../output")
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # just to satisfy grouper vis
    lines = grouper.run(boxes, img)

    cropper = LineCropper(padding=8, output_dir="../output")
    crops = cropper.crop_lines(img, lines)
    print(f"Produced {len(crops)} crops.")
