"""
Test Configuration and Shared Fixtures

Provides fixtures for testing the OCR pipeline frontend integration.
"""

import os
import sys
import tempfile

# Add project root to path BEFORE any other imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock torch to prevent import errors during test collection
# This allows tests to run without the full ML dependencies
class MockTorch:
    __version__ = "2.0.0"

sys.modules['torch'] = MockTorch()

# Now safe to import pytest and other test dependencies
import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock

# ─────────────────────────────────────────────────────────────────────────────
# Sample Data Constants
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_PIPELINE_OUTPUT = {
    "full_text": "Metformin 500mg BD\nTake after meals PO",
    "structured_json": {
        "lines": [
            {
                "line_id": 1,
                "text": "Metformin 500mg BD",
                "confidence": 0.92,
                "flagged": False,
            },
            {
                "line_id": 2,
                "text": "Take after meals PO",
                "confidence": 0.85,
                "flagged": False,
            },
        ]
    },
}

SAMPLE_PIPELINE_OUTPUT_EMPTY = {
    "full_text": "",
    "structured_json": {"lines": []},
}

SAMPLE_PIPELINE_OUTPUT_LOW_CONFIDENCE = {
    "full_text": "Unclear text",
    "structured_json": {
        "lines": [
            {
                "line_id": 1,
                "text": "Unclear text",
                "confidence": 0.45,
                "flagged": True,
            }
        ]
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Image Data
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_image():
    """Generate a synthetic BGR image for testing."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :100] = [255, 255, 255]  # Left half white
    img[:, 100:] = [200, 200, 200]  # Right half gray
    return img


@pytest.fixture
def sample_image_color():
    """Generate a synthetic color image for testing."""
    img = np.zeros((150, 300, 3), dtype=np.uint8)
    img[:50, :] = [255, 0, 0]      # Top: red
    img[50:100, :] = [0, 255, 0]    # Middle: green
    img[100:, :] = [0, 0, 255]      # Bottom: blue
    return img


@pytest.fixture
def sample_grayscale_image():
    """Generate a synthetic grayscale image for testing."""
    img = np.zeros((100, 200), dtype=np.uint8)
    img[:50, :] = 255
    img[50:, :] = 128
    return img


@pytest.fixture
def temp_image_file(sample_image):
    """Create a temporary image file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(path, sample_image)
    yield path
    if os.path.exists(path):
        os.remove(path)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Mock Objects
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def uploaded_file_mock(sample_image):
    """Create a mock Streamlit UploadedFile."""
    mock = MagicMock()
    mock.getvalue.return_value = cv2.imencode('.png', sample_image)[1].tobytes()
    mock.type = "image/png"
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Sample Results
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_result():
    """Return a sample OCR result dict."""
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


@pytest.fixture
def empty_result():
    """Return an empty OCR result dict."""
    return {
        "lines": [],
        "total_lines": 0,
        "avg_confidence": 0.0,
        "abbreviations_found": 0,
        "processing_time": "0s",
        "model": "TrOCR",
        "raw_text": "",
    }


@pytest.fixture
def single_line_result():
    """Return a single-line OCR result dict."""
    return {
        "lines": [
            {
                "line_id": 1,
                "text": "Aspirin 100mg",
                "expanded": "Aspirin 100 milligrams",
                "confidence": 0.95,
                "flagged": False,
                "abbreviations_in_line": 1,
            },
        ],
        "total_lines": 1,
        "avg_confidence": 0.95,
        "abbreviations_found": 1,
        "processing_time": "1.2s",
        "model": "TrOCR",
        "raw_text": "Aspirin 100mg",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: History
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_history():
    """Return an empty history list."""
    return []


@pytest.fixture
def sample_history():
    """Return a sample history list."""
    return [
        {
            "time": "10:30",
            "image_hash": "abc123",
            "preview": "Metformin 500mg BD",
            "confidence": 0.92,
            "line_count": 2,
        },
        {
            "time": "11:45",
            "image_hash": "def456",
            "preview": "Take after meals PO",
            "confidence": 0.85,
            "line_count": 1,
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Utility Values
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def image_hash():
    """Return a known image hash for testing."""
    return "abc12345"


@pytest.fixture
def max_history_size():
    """Return the max history size."""
    return 10


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Pipeline Mocks
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pipeline_run(monkeypatch):
    """Mock pipeline.main.run to return sample output."""
    def mock_run(*args, **kwargs):
        return SAMPLE_PIPELINE_OUTPUT["full_text"], SAMPLE_PIPELINE_OUTPUT["structured_json"]
    monkeypatch.setattr("frontend.pipeline_wrapper.pipeline_run", mock_run)
    return mock_run


@pytest.fixture
def mock_pipeline_run_empty(monkeypatch):
    """Mock pipeline.main.run to return empty output."""
    def mock_run(*args, **kwargs):
        return SAMPLE_PIPELINE_OUTPUT_EMPTY["full_text"], SAMPLE_PIPELINE_OUTPUT_EMPTY["structured_json"]
    monkeypatch.setattr("frontend.pipeline_wrapper.pipeline_run", mock_run)
    return mock_run


@pytest.fixture
def mock_pipeline_run_low_confidence(monkeypatch):
    """Mock pipeline.main.run to return low confidence output."""
    def mock_run(*args, **kwargs):
        return SAMPLE_PIPELINE_OUTPUT_LOW_CONFIDENCE["full_text"], SAMPLE_PIPELINE_OUTPUT_LOW_CONFIDENCE["structured_json"]
    monkeypatch.setattr("frontend.pipeline_wrapper.pipeline_run", mock_run)
    return mock_run