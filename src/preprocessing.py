import cv2
import numpy as np
import logging
from typing import Tuple

class Preprocessor:
    """
    Converts a raw full-page handwritten image into a clean binary image
    ready for line segmentation.
    Steps: grayscale → denoise → adaptive threshold → deskew
    """

    def __init__(
        self,
        blur_kernel: Tuple[int, int] = (3, 3),
        block_size: int = 25,
        c_constant: int = 10,
        deskew: bool = True,
    ):
        self.blur_kernel = blur_kernel
        self.block_size = block_size
        self.c_constant = c_constant
        self.deskew = deskew
        self._log = logging.getLogger(self.__class__.__name__)

    def process(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (binary_image, original_gray) both as numpy arrays (H, W).
        """
        self._log.info(f"Loading image: {image_path}")
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image at: {image_path}")

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self._log.debug(f"  Grayscale shape: {gray.shape}")

        denoised = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        binary = cv2.adaptiveThreshold(
            denoised,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=self.block_size,
            C=self.c_constant,
        )
        self._log.debug("  Adaptive thresholding applied")

        binary = self._remove_noise(binary)

        if self.deskew:
            angle = self._estimate_skew(binary)
            self._log.info(f"  Estimated skew angle: {angle:.2f}°")
            binary = self._rotate(binary, angle)
            gray = self._rotate(gray, angle)

        return binary, gray

    def _remove_noise(self, binary: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        return opened

    def _estimate_skew(self, binary: np.ndarray) -> float:
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
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)