"""
Frontend Utility Functions

er functions for the Streamlit frontend.
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

# Add pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.medical.abbreviations import expand_text, expand_with_details, get_abbreviation_count


def get_image_hash(img: np.ndarray) -> str:
    """Generate short hash for image."""
    return hashlib.md5(img.tobytes()).hexdigest()[:8]


def load_image_from_upload(uploaded_file) -> np.ndarray:
    """Load uploaded file as numpy array."""
    bytes_data = uploaded_file.getvalue()
    nparr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def run_prescription_ocr(
    image: np.ndarray,
    recognizer: str = "TrOCR",
    use_abbreviations: bool = True,
    confidence_threshold: float = 0.75,
) -> Dict[str, Any]:
    """
    Run the prescription OCR pipeline.

    Args:
        image: Input image as numpy array
        recognizer: "TrOCR" or "HTRVT"
        use_abbreviations: Whether to expand medical abbreviations
        confidence_threshold: Confidence threshold for flagging

    Returns:
        Dict with results compatible with frontend
    """
    RECOGNIZER_MAP = {
        "TrOCR (Recommended)": "TrOCR",
        "HTR-VT (Backup)": "HTRVT",
    }
    mapped_recognizer = RECOGNIZER_MAP.get(recognizer, recognizer)

    from frontend.pipeline_wrapper import run_ocr_pipeline

    return run_ocr_pipeline(
        image,
        recognizer=mapped_recognizer,
        use_abbreviations=use_abbreviations,
        confidence_threshold=confidence_threshold,
    )


def get_confidence_color(confidence: float) -> str:
    """Get color based on confidence level."""
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.6:
        return "orange"
    else:
        return "red"


def format_confidence(confidence: float) -> str:
    """Format confidence as percentage string."""
    return f"{confidence * 100:.1f}%"


def highlight_drug_names(text: str, drug_list: List[str] = None) -> str:
    """
    Highlight drug names in text.
    In production, this would use NER to find drugs.
    """
    if not drug_list:
        return text

    for drug in drug_list:
        if drug.lower() in text.lower():
            text = text.replace(drug, f"**{drug}**")
        if drug.upper() in text.upper():
            text = text.replace(drug.upper(), f"**{drug.upper()}**")

    return text


def save_scan_to_history(
    history: List[Dict],
    result: Dict,
    image_hash: str,
    max_history: int = 10
) -> List[Dict]:
    """
    Add scan result to history.

    Args:
        history: Current history list
        result: Scan result dict
        image_hash: Hash of the image
        max_history: Maximum history size

    Returns:
        Updated history list
    """
    new_entry = {
        "time": datetime.now().strftime("%H:%M"),
        "image_hash": image_hash,
        "preview": result.get("raw_text", "")[:50],
        "confidence": result.get("avg_confidence", 0),
        "line_count": result.get("total_lines", 0),
    }

    history.append(new_entry)

    # Keep only last max_history entries
    if len(history) > max_history:
        history = history[-max_history:]

    return history


def export_to_json(result: Dict, filename: str = "prescription_result.json") -> bytes:
    """Export result to JSON bytes."""
    return json.dumps(result, indent=2).encode("utf-8")


def export_to_txt(result: Dict, filename: str = "prescription_result.txt") -> bytes:
    """Export result to TXT bytes."""
    lines = []
    lines.append("=" * 50)
    lines.append("PRESCRIPTION OCR RESULT")
    lines.append("=" * 50)
    lines.append("")

    for line in result.get("lines", []):
        lines.append(f"Line {line['line_id']}: {line['text']}")
        if line.get("expanded") != line.get("text"):
            lines.append(f"  → {line['expanded']}")
        lines.append(f"  Confidence: {line['confidence']:.0%}")
        if line.get("flagged"):
            lines.append("  ⚠️ Low confidence")
        lines.append("")

    lines.append("=" * 50)
    lines.append(f"Total Lines: {result.get('total_lines', 0)}")
    lines.append(f"Average Confidence: {result.get('avg_confidence', 0):.0%}")
    lines.append(f"Abbreviations Found: {result.get('abbreviations_found', 0)}")
    lines.append(f"Model: {result.get('model', 'Unknown')}")
    lines.append("=" * 50)

    return "\n".join(lines).encode("utf-8")


def export_to_csv(result: Dict, filename: str = "prescription_result.csv") -> bytes:
    """Export result to CSV bytes."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["Line ID", "Text", "Expanded", "Confidence", "Flagged", "Abbreviations"])

    # Data
    for line in result.get("lines", []):
        writer.writerow([
            line.get("line_id", ""),
            line.get("text", ""),
            line.get("expanded", ""),
            f"{line.get('confidence', 0):.2%}",
            "Yes" if line.get("flagged") else "No",
            line.get("abbreviations_in_line", 0),
        ])

    return output.getvalue().encode("utf-8")


def create_sample_result() -> Dict[str, Any]:
    """
    Create a sample result for demo/testing.
    This simulates what the pipeline would return.
    """
    return {
        "lines": [
            {
                "line_id": 1,
                "text": "Metformin 500mg BD",
                "expanded": "Metformin 500 milligrams twice daily",
                "confidence": 0.92,
                "flagged": False,
                "abbreviations_in_line": 1,
            },
            {
                "line_id": 2,
                "text": "Take after meals PO",
                "expanded": "Take after meals by mouth",
                "confidence": 0.85,
                "flagged": False,
                "abbreviations_in_line": 1,
            },
            {
                "line_id": 3,
                "text": "Paracetamol 650mg TDS PRN",
                "expanded": "Paracetamol 650 milligrams three times daily as needed",
                "confidence": 0.78,
                "flagged": False,
                "abbreviations_in_line": 3,
            },
            {
                "line_id": 4,
                "text": "For 5 days",
                "expanded": "For 5 days",
                "confidence": 0.65,
                "flagged": True,
                "abbreviations_in_line": 0,
            },
        ],
        "total_lines": 4,
        "avg_confidence": 0.80,
        "abbreviations_found": 5,
        "processing_time": "2.3s",
        "model": "TrOCR (Recommended)",
        "raw_text": "Metformin 500mg BD\nTake after meals PO\nParacetamol 650mg TDS PRN\nFor 5 days",
    }