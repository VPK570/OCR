import logging
import numpy as np
from typing import Tuple, List
from data_types import PipelineOutput

class Evaluator:
    """
    Computes CER and WER against a ground-truth string.
    Uses pure dynamic-programming edit distance — no external libs required.
    """

    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)

    def evaluate(
        self,
        hypothesis: str,
        reference: str,
        label: str = "",
    ) -> Tuple[float, float]:
        cer = self._char_error_rate(hypothesis, reference)
        wer = self._word_error_rate(hypothesis, reference)
        tag = f"[{label}] " if label else ""
        self._log.info(f"  {tag}CER={cer:.4f}  WER={wer:.4f}")
        return cer, wer

    def full_report(self, output: PipelineOutput) -> str:
        lines = [
            "=" * 60,
            "  EVALUATION REPORT",
            "=" * 60,
            f"  Lines decoded     : {len(output.ocr_results)}",
            f"  Avg OCR confidence: "
            f"{np.mean([r.confidence for r in output.ocr_results]):.3f}",
            "",
            "  ── Character Error Rate (CER) ──",
            f"    Before refinement : {output.cer_before:.4f}",
            f"    After  refinement : {output.cer_after:.4f}",
            f"    Δ CER             : {output.cer_before - output.cer_after:+.4f}",
            "",
            "  ── Word Error Rate (WER) ──",
            f"    Before refinement : {output.wer_before:.4f}",
            f"    After  refinement : {output.wer_after:.4f}",
            f"    Δ WER             : {output.wer_before - output.wer_after:+.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    @staticmethod
    def _edit_distance(a: List, b: List) -> int:
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = (
                    prev if a[i - 1] == b[j - 1]
                    else 1 + min(prev, dp[j], dp[j - 1])
                )
                prev = temp
        return dp[n]

    def _char_error_rate(self, hyp: str, ref: str) -> float:
        if not ref:
            return 0.0 if not hyp else 1.0
        dist = self._edit_distance(list(hyp), list(ref))
        return dist / len(ref)

    def _word_error_rate(self, hyp: str, ref: str) -> float:
        ref_words = ref.split()
        hyp_words = hyp.split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        dist = self._edit_distance(hyp_words, ref_words)
        return dist / len(ref_words)
