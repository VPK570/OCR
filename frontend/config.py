"""
Configuration settings for the Prescription OCR Frontend
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(BASE_DIR, "pipeline")
CRAFT_DIR = os.path.join(BASE_DIR, "CRAFT-pytorch")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# CRAFT weights
CRAFT_WEIGHTS = os.path.join(CRAFT_DIR, "weights", "craft_mlt_25k.pth")
HTRVT_WEIGHTS = os.path.join(WEIGHTS_DIR, "best_CER.pth")

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_CONFIG = {
    "CONFIDENCE_THRESHOLD": 0.75,
    "DEBUG": False,
    "USE_CLAHE": True,
    "DESKEW": False,
    "CUDA": False,  # Set to True if you have GPU
    "TEXT_THRESHOLD": 0.7,
    "LINK_THRESHOLD": 0.4,
    "LOW_TEXT": 0.4,
    "MIN_BOX_AREA": 100,
    "IOU_THRESHOLD": 0.5,
    "CROP_PADDING": 8,
}

# ─────────────────────────────────────��───────────────────────────────────────
# FRONTEND SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

# Recognizer options
RECOGNIZERS = {
    "TrOCR (Recommended)": "TrOCR",
    "HTR-VT (Backup)": "HTRVT",
}

# Theme settings
DEFAULT_THEME = "light"  # "light" or "dark"

# UI Settings
MAX_IMAGE_SIZE_MB = 10
HISTORY_LIMIT = 10
AUTO_EXPAND_ABBREVIATIONS = True

# ─────────────────────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────────────────────

# Drug highlighting colors (hex)
DRUG_COLORS = {
    "drug_name": "#FF6B6B",      # Red for drug names
    "dosage": "#4ECDC4",          # Teal for dosages
    "frequency": "#45B7D1",       # Blue for frequency
    "route": "#96CEB4",           # Green for route
    "abbreviation": "#FFEAA7",   # Yellow for abbreviations
}

# Confidence colors
CONFIDENCE_COLORS = {
    "high": "#2ECC71",           # Green (>0.8)
    "medium": "#F39C12",         # Orange (0.6-0.8)
    "low": "#E74C3C",            # Red (<0.6)
}

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

EXPORT_FORMATS = ["JSON", "TXT", "CSV"]
DEFAULT_EXPORT_FORMAT = "JSON"

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_confidence_color(confidence: float) -> str:
    """Get color for confidence score."""
    if confidence >= 0.8:
        return CONFIDENCE_COLORS["high"]
    elif confidence >= 0.6:
        return CONFIDENCE_COLORS["medium"]
    else:
        return CONFIDENCE_COLORS["low"]


def get_confidence_label(confidence: float) -> str:
    """Get label for confidence score."""
    if confidence >= 0.8:
        return "High"
    elif confidence >= 0.6:
        return "Medium"
    else:
        return "Low"