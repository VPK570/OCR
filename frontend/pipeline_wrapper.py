"""
Pipeline Wrapper for Frontend

Bridges Streamlit uploads (numpy arrays) to the OCR pipeline (file paths).
"""

import os
import sys
import tempfile
import uuid
from typing import Dict, Any
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.main import CONFIG, run as pipeline_run
from pipeline.medical.abbreviations import expand_text, get_abbreviation_count


def run_ocr_pipeline(
    image: np.ndarray,
    recognizer: str = "TrOCR",
    use_abbreviations: bool = True,
    confidence_threshold: float = 0.75,
) -> Dict[str, Any]:
    """
    Run the OCR pipeline on a numpy image.
    
    Args:
        image: Input image as numpy array (BGR format from cv2)
        recognizer: "TrOCR" or "HTRVT"
        use_abbreviations: Whether to expand medical abbreviations
        confidence_threshold: Confidence below which lines are flagged
    
    Returns:
        Unified dict with lines, avg_confidence, abbreviations_found, etc.
    """
    temp_dir = tempfile.gettempdir()
    temp_filename = f"ocr_temp_{uuid.uuid4().hex[:8]}.png"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    try:
        cv2.imwrite(temp_path, image)
        
        cfg = dict(CONFIG)

        RECOGNIZER_MAP = {
            "TrOCR (Recommended)": "TrOCR",
            "HTR-VT (Backup)": "HTRVT",
        }
        cfg["USE_RECOGNIZER"] = RECOGNIZER_MAP.get(recognizer, recognizer)
        cfg["CONFIDENCE_THRESHOLD"] = confidence_threshold
        cfg["DEBUG"] = False
        
        full_text, structured_json = pipeline_run(temp_path, cfg)
        
        result = {
            "lines": [],
            "total_lines": 0,
            "avg_confidence": 0.0,
            "abbreviations_found": 0,
            "processing_time": "0s",
            "model": recognizer,
            "raw_text": full_text,
        }
        
        if structured_json and "lines" in structured_json:
            total_confidence = 0.0
            all_lines = []
            
            for line in structured_json["lines"]:
                text = line.get("text", "")
                confidence = line.get("confidence", 0.0)
                flagged = line.get("flagged", confidence < confidence_threshold)
                
                if use_abbreviations and text:
                    expanded = expand_text(text)
                else:
                    expanded = text
                
                abbrev_count = get_abbreviation_count(text)
                
                all_lines.append({
                    "line_id": line.get("line_id", 0),
                    "text": text,
                    "expanded": expanded,
                    "confidence": confidence,
                    "flagged": flagged,
                    "abbreviations_in_line": abbrev_count,
                })
                
                total_confidence += confidence
                result["abbreviations_found"] += abbrev_count
            
            result["lines"] = all_lines
            result["total_lines"] = len(all_lines)
            result["avg_confidence"] = (
                total_confidence / len(all_lines) if all_lines else 0.0
            )
        
        return result
        
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass