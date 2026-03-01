"""
preprocess.py — Stage 1: Image Preprocessing

Converts a raw full-page handwritten image into a clean grayscale image
ready for CRAFT text detection.

Steps: load → grayscale → denoise → optional CLAHE → optional deskew

IMPORTANT: We do NOT threshold here. CRAFT needs a proper grayscale image.
"""

import cv2
import numpy as np
import os
from typing import Tuple

from utils import get_logger, load_image, setup_output_dirs


class Preprocessor:
    """
    Stage 1 of the pipeline.

    Args:
        blur_kernel  : Gaussian blur kernel size (must be odd). Use (0,0) to skip.
        use_clahe    : Apply CLAHE contrast enhancement before returning.
        deskew       : Estimate and correct page skew via Hough transform.
        output_dir   : If set, saves the preprocessed image here.
    """

    def __init__(
        self,
        blur_kernel: Tuple[int, int] = (3, 3),
        use_clahe: bool = True,
        deskew: bool = True,
        output_dir: str = None,
    ):
        self.blur_kernel = blur_kernel
        self.use_clahe = use_clahe
        self.deskew = deskew
        self.output_dir = output_dir
        self._log = get_logger("Preprocessor")

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def process(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the full preprocessing chain.

        Returns:
            gray    : Processed grayscale image (H, W) — fed into CRAFT
            original: Original BGR image (H, W, 3)  — used for line cropping later
        """
        self._log.info(f"Loading image: {image_path}")
        original_bgr = load_image(image_path)
        h, w = original_bgr.shape[:2]
        self._log.info(f"  Image size: {w}×{h}")

        # Step 1 — Grayscale
        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        self._log.debug("  ✓ Grayscale conversion")

        # Step 2 — Denoise
        if self.blur_kernel != (0, 0):
            gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)
            self._log.debug(f"  ✓ Gaussian blur {self.blur_kernel}")

        # Step 3 — CLAHE contrast enhancement
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            self._log.debug("  ✓ CLAHE applied")

        # Step 4 — Deskew
        if self.deskew:
            angle = self._estimate_skew(gray)
            self._log.info(f"  Estimated skew: {angle:.2f}°")
            if abs(angle) > 0.3:
                gray = self._rotate(gray, angle)
                original_bgr = self._rotate(original_bgr, angle)
                self._log.debug(f"  ✓ Rotated by {angle:.2f}°")
            else:
                self._log.debug("  Skew within tolerance — no rotation applied")

        # Save debug output
        if self.output_dir:
            save_path = os.path.join(self.output_dir, "01_preprocessed", "gray.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, gray)
            self._log.info(f"  Saved preprocessed image → {save_path}")

        return gray, original_bgr

    # ─────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────

    def _estimate_skew(self, gray: np.ndarray) -> float:
        """
        Estimates page skew angle using minAreaRect on text pixel coordinates.
        Returns angle in degrees (positive = rotate CCW to correct).
        """
        # Threshold just for skew estimation (not for pipeline output)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(binary > 0))
        if coords.shape[0] < 10:
            return 0.0
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        return -angle

    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotates image around its center without cropping."""
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )


# ─────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "../sample_images/image.png"
    p = Preprocessor(use_clahe=True, deskew=True, output_dir="../output")
    gray, bgr = p.process(img_path)
    print(f"gray shape: {gray.shape}, bgr shape: {bgr.shape}")
    cv2.imwrite("/tmp/preprocessed_test.png", gray)
    print("Saved test result → /tmp/preprocessed_test.png")
