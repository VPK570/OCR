"""
recognizer/base.py — Abstract base class for all OCR recognizers.

Both HTRVTRecognizer and TrOCRRecognizer implement this interface,
making the pipeline swappable via config with zero changes elsewhere.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseRecognizer(ABC):
    """
    Abstract OCR recognizer.

    Every concrete recognizer must implement `predict()`.
    The `predict_batch()` default loops `predict()` — override for efficiency.
    """

    @abstractmethod
    def predict(self, image_crop: np.ndarray) -> Tuple[str, float]:
        """
        Recognize text in a single line crop.

        Args:
            image_crop : BGR or grayscale numpy array of one text line.

        Returns:
            text       : Decoded text string.
            confidence : Float in [0, 1]. Higher = more confident.
        """
        ...

    def predict_batch(
        self, crops: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        """
        Recognize text in a list of crops.
        Default implementation loops over predict() — override for batching.
        """
        return [self.predict(crop) for crop in crops]
