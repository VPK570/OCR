"""
postprocess.py — Stages 7–9: Confidence Scoring, Text Reconstruction, Output Writing

Takes per-line recognition results and:
  - Flags low-confidence lines
  - Reconstructs full text (top-to-bottom)
  - Writes structured JSON and plain text
  - Saves annotated crop images with confidence overlays
  - Copies flagged crops to error_analysis/

Writes:
  06_recognition_raw/results_raw.json
  07_recognition_with_confidence/line_XX_annotated.png
  08_final_text/output.txt
  08_final_text/output.json
  09_error_analysis/line_XX_flagged.png   (low-confidence lines only)
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

from utils import get_logger, save_json, save_text, overlay_text_on_crop


LineResult = Tuple[int, str, float, np.ndarray]  # (line_id, text, confidence, crop)


class TextReconstructor:
    """
    Stages 7–9 of the pipeline.

    Args:
        confidence_threshold : Lines below this are flagged for review.
        output_dir           : Root output directory.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        output_dir: str = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.output_dir = output_dir
        self._log = get_logger("TextReconstructor")

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def reconstruct(
        self,
        line_results: List[LineResult],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Sorts results top-to-bottom, joins text, builds structured JSON.

        Args:
            line_results : List of (line_id, text, confidence, crop_image).

        Returns:
            full_text       : Newline-joined recognized text.
            structured_json : Dict with per-line details.
        """
        # Sort by line_id (guaranteed top-to-bottom from cropper)
        sorted_results = sorted(line_results, key=lambda r: r[0])

        lines_json = []
        full_lines = []
        flagged_count = 0

        for line_id, text, confidence, crop in sorted_results:
            flagged = confidence < self.confidence_threshold
            if flagged:
                flagged_count += 1
                self._log.warning(
                    f"  ⚠ Line {line_id:02d} flagged (conf={confidence:.3f}): {text!r}"
                )
            else:
                self._log.info(
                    f"  Line {line_id:02d} [conf={confidence:.3f}]: {text!r}"
                )

            full_lines.append(text)
            lines_json.append({
                "line_id": line_id,
                "text": text,
                "confidence": round(confidence, 4),
                "flagged": flagged,
            })

        full_text = "\n".join(full_lines)
        structured_json = {"lines": lines_json}

        self._log.info(
            f"  Reconstructed {len(sorted_results)} lines "
            f"({flagged_count} flagged below conf={self.confidence_threshold})"
        )

        if self.output_dir:
            self._save_all(sorted_results, full_text, structured_json)

        return full_text, structured_json

    # ─────────────────────────────────────────────
    # Internal Save Methods
    # ─────────────────────────────────────────────

    def _save_all(
        self,
        sorted_results: List[LineResult],
        full_text: str,
        structured_json: Dict[str, Any],
    ) -> None:

        raw_dir   = os.path.join(self.output_dir, "06_recognition_raw")
        conf_dir  = os.path.join(self.output_dir, "07_recognition_with_confidence")
        text_dir  = os.path.join(self.output_dir, "08_final_text")
        error_dir = os.path.join(self.output_dir, "09_error_analysis")

        for d in [raw_dir, conf_dir, text_dir, error_dir]:
            os.makedirs(d, exist_ok=True)

        # ── 06: Raw recognition JSON ──
        save_json(structured_json, os.path.join(raw_dir, "results_raw.json"))
        self._log.info(f"  Saved raw results → {raw_dir}/results_raw.json")

        # ── 07 & 09: Annotated crop images ──
        for line_id, text, confidence, crop in sorted_results:
            flagged = confidence < self.confidence_threshold
            annotated = overlay_text_on_crop(crop, text, confidence, flagged)

            # Save to 07
            conf_path = os.path.join(conf_dir, f"line_{line_id:02d}_annotated.png")
            cv2.imwrite(conf_path, annotated)

            # Also copy to 09 if flagged
            if flagged:
                err_path = os.path.join(error_dir, f"line_{line_id:02d}_flagged.png")
                cv2.imwrite(err_path, annotated)

        self._log.info(f"  Saved annotated crops → {conf_dir}")

        # ── 08: Final text + JSON ──
        save_text(full_text, os.path.join(text_dir, "output.txt"))
        save_json(structured_json, os.path.join(text_dir, "output.json"))
        self._log.info(f"  Saved final output → {text_dir}")
