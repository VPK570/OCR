import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class LineRegion:
    y_start: int
    y_end: int
    x_start: int
    x_end: int
    image_crop: np.ndarray

    @property
    def height(self) -> int:
        return self.y_end - self.y_start

    @property
    def width(self) -> int:
        return self.x_end - self.x_start


@dataclass
class OCRResult:
    line_index: int
    raw_text: str
    confidence: float
    region: LineRegion


@dataclass
class PipelineOutput:
    raw_text: str
    refined_text: str
    ocr_results: List[OCRResult]
    cer_before: float
    cer_after: float
    wer_before: float
    wer_after: float
    metadata: Dict[str, Any] = field(default_factory=dict)
