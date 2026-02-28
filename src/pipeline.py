import logging
from typing import Optional

from data_types import PipelineOutput
from preprocessing import Preprocessor
from segmentation import LineSegmenter
from ocr import OCRModel
from refinement import TextRefiner
from evaluation import Evaluator

class HandwritingDigitizationPipeline:
    """
    Top-level orchestrator.
    Wires: Preprocessor → LineSegmenter → OCRModel → TextRefiner → Evaluator
    """

    def __init__(
        self,
        preprocessor: Preprocessor,
        segmenter: LineSegmenter,
        ocr: OCRModel,
        refiner: TextRefiner,
        evaluator: Evaluator,
    ):
        self.preprocessor = preprocessor
        self.segmenter = segmenter
        self.ocr = ocr
        self.refiner = refiner
        self.evaluator = evaluator
        self._log = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        image_path: str,
        ground_truth: Optional[str] = None,
    ) -> PipelineOutput:
        self._log.info("━" * 60)
        self._log.info(" STAGE 1 — Preprocessing")
        self._log.info("━" * 60)
        binary, gray = self.preprocessor.process(image_path)
        self._log.info(f"  Output shape: {binary.shape}")

        self._log.info("━" * 60)
        self._log.info(" STAGE 2 — Line Segmentation")
        self._log.info("━" * 60)
        # Support both standard and CRAFT segmenters
        if hasattr(self.segmenter, 'segment'):
            import inspect
            sig = inspect.signature(self.segmenter.segment)
            if 'image_path' in sig.parameters:
                regions = self.segmenter.segment(image_path, gray)
            else:
                regions = self.segmenter.segment(binary, gray)
        
        self._log.info(f"  Total lines: {len(regions)}")

        self._log.info("━" * 60)
        self._log.info(" STAGE 3 — OCR")
        self._log.info("━" * 60)
        ocr_results = self.ocr.recognize_lines(regions)

        raw_text = "\n".join(r.raw_text for r in ocr_results)
        self._log.info("\n  ── RAW OCR TEXT ──")
        for line in raw_text.splitlines():
            self._log.info(f"  {line}")

        self._log.info("━" * 60)
        self._log.info(" STAGE 4 — Text Refinement (DistilBERT)")
        self._log.info("━" * 60)
        refined_text = self.refiner.refine(ocr_results)
        self._log.info("\n  ── REFINED TEXT ──")
        for line in refined_text.splitlines():
            self._log.info(f"  {line}")

        cer_before = cer_after = wer_before = wer_after = float("nan")
        if ground_truth is not None:
            self._log.info("━" * 60)
            self._log.info(" STAGE 5 — Evaluation")
            self._log.info("━" * 60)
            cer_before, wer_before = self.evaluator.evaluate(
                raw_text, ground_truth, label="before refinement"
            )
            cer_after, wer_after = self.evaluator.evaluate(
                refined_text, ground_truth, label="after  refinement"
            )

        output = PipelineOutput(
            raw_text=raw_text,
            refined_text=refined_text,
            ocr_results=ocr_results,
            cer_before=cer_before,
            cer_after=cer_after,
            wer_before=wer_before,
            wer_after=wer_after,
            metadata={
                "image_path": image_path,
                "num_lines": len(regions),
                "binary_shape": binary.shape,
            },
        )
        return output
