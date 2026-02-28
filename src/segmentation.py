import cv2
import numpy as np
import logging
from typing import List, Tuple
from data_types import LineRegion

class LineSegmenter:
    """
    Detects text lines via horizontal projection profile analysis.
    Adaptive gap detection — no hardcoded pixel thresholds.
    """

    def __init__(
        self,
        min_line_height_ratio: float = 0.005,
        valley_threshold_percentile: float = 15.0,
        dilation_iterations: int = 2,
        padding: int = 4,
    ):
        self.min_line_height_ratio = min_line_height_ratio
        self.valley_threshold_percentile = valley_threshold_percentile
        self.dilation_iterations = dilation_iterations
        self.padding = padding
        self._log = logging.getLogger(self.__class__.__name__)

    def segment(
        self, binary: np.ndarray, original_gray: np.ndarray
    ) -> List[LineRegion]:
        h, w = binary.shape

        dilated = self._dilate_horizontal(binary)

        projection = np.sum(dilated, axis=1).astype(np.float32)
        self._log.debug(f"  Projection profile: min={projection.min():.0f}, "
                        f"max={projection.max():.0f}")

        valley_thresh = np.percentile(
            projection[projection > 0], self.valley_threshold_percentile
        ) if np.any(projection > 0) else 0.0

        self._log.info(f"  Valley threshold (p{self.valley_threshold_percentile:.0f}): "
                       f"{valley_thresh:.1f}")

        line_mask = projection > valley_thresh
        line_bands = self._mask_to_bands(line_mask)

        min_h = int(h * self.min_line_height_ratio)
        line_bands = [(s, e) for s, e in line_bands if (e - s) >= min_h]

        self._log.info(f"  Detected {len(line_bands)} line bands")

        regions: List[LineRegion] = []
        for idx, (y0, y1) in enumerate(line_bands):
            y0p = max(0, y0 - self.padding)
            y1p = min(h, y1 + self.padding)
            strip_bin = binary[y0p:y1p, :]

            col_proj = np.sum(strip_bin, axis=0)
            nonzero_cols = np.where(col_proj > 0)[0]
            x0 = int(nonzero_cols.min()) if len(nonzero_cols) else 0
            x1 = int(nonzero_cols.max()) if len(nonzero_cols) else w
            x0 = max(0, x0 - self.padding)
            x1 = min(w, x1 + self.padding)

            crop_gray = original_gray[y0p:y1p, x0:x1]
            regions.append(
                LineRegion(
                    y_start=y0p, y_end=y1p,
                    x_start=x0, x_end=x1,
                    image_crop=crop_gray,
                )
            )
            self._log.debug(
                f"  Line {idx:03d}: y=[{y0p},{y1p}] x=[{x0},{x1}] "
                f"size={crop_gray.shape}"
            )

        return regions

    def _dilate_horizontal(self, binary: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        return cv2.dilate(binary, kernel, iterations=self.dilation_iterations)

    @staticmethod
    def _mask_to_bands(mask: np.ndarray) -> List[Tuple[int, int]]:
        bands = []
        in_band = False
        start = 0
        for i, val in enumerate(mask):
            if val and not in_band:
                start = i
                in_band = True
            elif not val and in_band:
                bands.append((start, i))
                in_band = False
        if in_band:
            bands.append((start, len(mask)))
        return bands
