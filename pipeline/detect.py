"""
detect.py — Stage 2: CRAFT Text Detection

Loads the CRAFT model from the existing CRAFT-pytorch repo (no code copy).
Runs detection on a grayscale image and returns axis-aligned bounding boxes.

Writes:
  02_detection_raw/boxes.json       — raw polygon data
  03_detection_visualized/det.png   — boxes drawn on image
"""

import sys
import os
import json
import time
import cv2
import numpy as np
import torch
from collections import OrderedDict
from typing import List, Tuple

from utils import get_logger, save_json, draw_boxes


class CRAFTDetector:
    """
    Stage 2 of the pipeline.

    Wraps the CRAFT-pytorch repo. Adds the repo to sys.path so we can
    import directly from the original codebase without duplicating code.

    Args:
        craft_dir     : Absolute path to the CRAFT-pytorch directory.
        weights_path  : Path to craft_mlt_25k.pth weights file.
        cuda          : Use GPU if True (CPU fallback if unavailable).
        text_threshold: CRAFT text confidence threshold (default 0.7).
        link_threshold: CRAFT link confidence threshold (default 0.4).
        low_text      : Lower-bound text score (default 0.4).
        canvas_size   : Image size for CRAFT inference (default 1280).
        mag_ratio     : Magnification ratio (default 1.5).
    """

    def __init__(
        self,
        craft_dir: str,
        weights_path: str,
        cuda: bool = False,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
        canvas_size: int = 1280,
        mag_ratio: float = 1.5,
        output_dir: str = None,
    ):
        self._log = get_logger("CRAFTDetector")
        self.cuda = cuda and torch.cuda.is_available()
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.output_dir = output_dir

        # ── Add CRAFT-pytorch to import path ──
        craft_dir = os.path.abspath(craft_dir)
        if craft_dir not in sys.path:
            sys.path.insert(0, craft_dir)

        # ── Import from CRAFT-pytorch ──
        from craft import CRAFT
        import craft_utils as cu
        import imgproc as ip

        self._craft_utils = cu
        self._imgproc = ip

        # ── Load model ──
        self._log.info(f"Loading CRAFT model from {weights_path} ...")
        self.net = CRAFT()
        map_loc = "cuda" if self.cuda else "cpu"
        state = torch.load(weights_path, map_location=map_loc)
        self.net.load_state_dict(self._copy_state_dict(state))
        if self.cuda:
            self.net = self.net.cuda()
        self.net.eval()
        self._log.info(f"  CRAFT ready. Device: {'GPU' if self.cuda else 'CPU'}")

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def detect(self, gray_image: np.ndarray) -> List[np.ndarray]:
        """
        Runs CRAFT on a grayscale image.

        CRAFT also accepts RGB — convert internally.

        Returns:
            polys : List of polygon arrays (each polygon is Nx2 array of points).
        """
        # CRAFT expects RGB-style 3-channel
        rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

        t0 = time.time()
        img_resized, target_ratio, _ = self._imgproc.resize_aspect_ratio(
            rgb, self.canvas_size,
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=self.mag_ratio
        )
        ratio_h = ratio_w = 1 / target_ratio

        x = self._imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        if self.cuda:
            x = x.cuda()

        with torch.no_grad():
            y, _ = self.net(x)

        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        boxes, polys = self._craft_utils.getDetBoxes(
            score_text, score_link,
            self.text_threshold, self.link_threshold,
            self.low_text, poly=False
        )
        boxes = self._craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = self._craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

        # Replace None polys with corresponding box
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        elapsed = time.time() - t0
        self._log.info(f"  CRAFT detected {len(polys)} regions in {elapsed:.2f}s")
        return polys

    def polygons_to_rects(
        self, polys: List[np.ndarray]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Converts CRAFT polygon arrays to axis-aligned (x1, y1, x2, y2) rects.
        """
        rects = []
        for poly in polys:
            if poly is None:
                continue
            pts = np.array(poly)
            x1 = int(np.min(pts[:, 0]))
            y1 = int(np.min(pts[:, 1]))
            x2 = int(np.max(pts[:, 0]))
            y2 = int(np.max(pts[:, 1]))
            rects.append((x1, y1, x2, y2))
        return rects

    def run(
        self,
        gray_image: np.ndarray,
        image_name: str = "image",
    ) -> List[Tuple[int, int, int, int]]:
        """
        Full detect → convert → save cycle.

        Returns list of (x1, y1, x2, y2) bounding boxes.
        """
        polys = self.detect(gray_image)
        rects = self.polygons_to_rects(polys)
        self._log.info(f"  Converted to {len(rects)} bounding rects")

        if self.output_dir:
            # ── Save raw boxes JSON ──
            json_path = os.path.join(self.output_dir, "02_detection_raw", "boxes.json")
            boxes_data = [
                {"box_id": i, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
                for i, (x1, y1, x2, y2) in enumerate(rects)
            ]
            save_json({"image": image_name, "boxes": boxes_data}, json_path)
            self._log.info(f"  Saved raw boxes → {json_path}")

            # ── Save visualization ──
            vis = draw_boxes(gray_image, rects, color=(0, 200, 50))
            vis_path = os.path.join(
                self.output_dir, "03_detection_visualized", "det.png"
            )
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
            cv2.imwrite(vis_path, vis)
            self._log.info(f"  Saved detection visualization → {vis_path}")

        return rects

    # ─────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _copy_state_dict(state_dict):
        """Removes 'module.' prefix from DataParallel-saved weights."""
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state[name] = v
        return new_state


# ─────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys as _sys
    img_path = _sys.argv[1] if len(_sys.argv) > 1 else "../sample_images/image.png"

    detector = CRAFTDetector(
        craft_dir="../CRAFT-pytorch",
        weights_path="../CRAFT-pytorch/weights/craft_mlt_25k.pth",
        cuda=False,
        output_dir="../output",
    )

    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    rects = detector.run(gray, image_name=os.path.basename(img_path))
    print(f"Detected {len(rects)} boxes.")
    for i, r in enumerate(rects[:5]):
        print(f"  Box {i}: {r}")
