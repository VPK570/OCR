import logging
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Optional
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from data_types import LineRegion, OCRResult

class OCRModel:
    """
    Wraps microsoft/trocr-base-handwritten.
    Produces text and a proxy confidence score from decoder log-probs.
    """

    MODEL_ID = "microsoft/trocr-base-handwritten"

    def __init__(self, device: Optional[str] = None, batch_size: int = 4):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.info(f"Loading TrOCR ({self.MODEL_ID}) on {self.device} …")
        self.processor = TrOCRProcessor.from_pretrained(self.MODEL_ID)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_ID)
        self.model.to(self.device)
        self.model.eval()
        self._log.info("  TrOCR ready.")

    @torch.no_grad()
    def recognize_lines(self, regions: List[LineRegion]) -> List[OCRResult]:
        results: List[OCRResult] = []
        for batch_start in range(0, len(regions), self.batch_size):
            batch = regions[batch_start: batch_start + self.batch_size]
            pil_images = [
                Image.fromarray(r.image_crop).convert("RGB") for r in batch
            ]
            pixel_values = self.processor(
                images=pil_images, return_tensors="pt"
            ).pixel_values.to(self.device)

            outputs = self.model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=128,
            )

            decoded = self.processor.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )

            confidences = self._compute_confidence(outputs)

            for i, (region, text, conf) in enumerate(
                zip(batch, decoded, confidences)
            ):
                line_idx = batch_start + i
                self._log.info(
                    f"  Line {line_idx:03d} [conf={conf:.3f}]: {text!r}"
                )
                results.append(
                    OCRResult(
                        line_index=line_idx,
                        raw_text=text,
                        confidence=conf,
                        region=region,
                    )
                )
        return results

    def _compute_confidence(self, outputs) -> List[float]:
        if not hasattr(outputs, "scores") or outputs.scores is None:
            return [1.0] * outputs.sequences.shape[0]
        scores_stack = torch.stack(outputs.scores, dim=1)
        log_probs = F.log_softmax(scores_stack, dim=-1)
        token_ids = outputs.sequences[:, 1: 1 + scores_stack.shape[1]]
        token_ids = token_ids.unsqueeze(-1)
        gathered = log_probs.gather(-1, token_ids).squeeze(-1)
        mean_log_prob = gathered.mean(dim=-1)
        confidences = mean_log_prob.exp().clamp(0.0, 1.0).cpu().tolist()
        return confidences
