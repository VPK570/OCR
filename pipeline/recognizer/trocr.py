"""
recognizer/trocr.py — TrOCR recognizer

Uses microsoft/trocr-base-handwritten via HuggingFace Transformers.
Downloads once, caches locally. No repo clone needed.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Tuple

# Add pipeline root to path so we can import base
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from recognizer.base import BaseRecognizer
from utils import get_logger


class TrOCRRecognizer(BaseRecognizer):
    """
    Wraps microsoft/trocr-base-handwritten.

    Args:
        model_id : HuggingFace model ID (default: microsoft/trocr-base-handwritten).
        device   : 'cpu', 'cuda', or None for auto-detection.
    """

    MODEL_ID = "microsoft/trocr-base-handwritten"

    def __init__(
        self,
        model_id: str = None,
        device: Optional[str] = None,
    ):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._log = get_logger("TrOCRRecognizer")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_id = model_id or self.MODEL_ID

        self._log.info(f"Loading TrOCR ({model_id}) on {self.device} ...")
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self._log.info("  TrOCR ready.")

    @torch.no_grad()
    def predict(self, image_crop: np.ndarray) -> Tuple[str, float]:
        """
        Runs TrOCR on a single line crop.

        Returns:
            (text, confidence)
        """
        # Convert to RGB PIL image
        if len(image_crop.shape) == 2:
            pil_img = Image.fromarray(image_crop).convert("RGB")
        else:
            import cv2
            pil_img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))

        pixel_values = self.processor(
            images=pil_img, return_tensors="pt"
        ).pixel_values.to(self.device)

        outputs = self.model.generate(
            pixel_values,
            num_beams=5,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=128,
        )

        text = self.processor.decode(
            outputs.sequences[0], skip_special_tokens=True
        )
        confidence = self._compute_confidence(outputs)
        return text, confidence

    def _compute_confidence(self, outputs) -> float:
        """Mean exponential of max log-prob at each decoder step."""
        if not hasattr(outputs, "scores") or outputs.scores is None:
            return 1.0
        scores_stack = torch.stack(outputs.scores, dim=1)         # [1, T, vocab]
        log_probs = F.log_softmax(scores_stack, dim=-1)
        token_ids = outputs.sequences[:, 1: 1 + scores_stack.shape[1]]
        token_ids = token_ids.unsqueeze(-1)
        gathered = log_probs.gather(-1, token_ids).squeeze(-1)
        mean_log = gathered.mean(dim=-1)
        return float(mean_log.exp().clamp(0.0, 1.0).item())
