"""
Tests for frontend/utils.py

Tests all utility functions in the frontend utils module.
"""

import os
import sys
import json
import pytest
import numpy as np
import cv2
from io import BytesIO
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frontend.utils import (
    get_image_hash,
    load_image_from_upload,
    get_confidence_color,
    format_confidence,
    highlight_drug_names,
    save_scan_to_history,
    export_to_json,
    export_to_txt,
    export_to_csv,
)


class TestGetImageHash:
    """Tests for get_image_hash function."""
    
    def test_same_image_same_hash(self, sample_image):
        """Same image should produce the same hash."""
        hash1 = get_image_hash(sample_image)
        hash2 = get_image_hash(sample_image)
        assert hash1 == hash2
    
    def test_different_images_different_hash(self, sample_image, sample_image_color):
        """Different images should produce different hashes."""
        hash1 = get_image_hash(sample_image)
        hash2 = get_image_hash(sample_image_color)
        assert hash1 != hash2
    
    def test_hash_length(self, sample_image):
        """Hash should be 8 characters."""
        hash_val = get_image_hash(sample_image)
        assert len(hash_val) == 8
    
    def test_hash_is_string(self, sample_image):
        """Hash should be a string."""
        hash_val = get_image_hash(sample_image)
        assert isinstance(hash_val, str)


class TestLoadImageFromUpload:
    """Tests for load_image_from_upload function."""
    
    def test_load_valid_image(self, uploaded_file_mock):
        """Should load a valid image from uploaded file."""
        img = load_image_from_upload(uploaded_file_mock)
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 3  # BGR color image
        assert img.shape[2] == 3
    
    def test_load_returns_bgr_format(self, uploaded_file_mock, sample_image):
        """Loaded image should be in BGR format."""
        img = load_image_from_upload(uploaded_file_mock)
        # Verify it's BGR by checking the first pixel
        expected = sample_image[0, 0]
        actual = img[0, 0]
        np.testing.assert_array_equal(actual, expected)


class TestGetConfidenceColor:
    """Tests for get_confidence_color function."""
    
    def test_high_confidence_returns_green(self):
        """Confidence >= 0.8 should return 'green'."""
        assert get_confidence_color(0.8) == "green"
        assert get_confidence_color(0.85) == "green"
        assert get_confidence_color(0.95) == "green"
        assert get_confidence_color(1.0) == "green"
    
    def test_medium_confidence_returns_orange(self):
        """Confidence 0.6-0.79 should return 'orange'."""
        assert get_confidence_color(0.6) == "orange"
        assert get_confidence_color(0.65) == "orange"
        assert get_confidence_color(0.75) == "orange"
        assert get_confidence_color(0.79) == "orange"
    
    def test_low_confidence_returns_red(self):
        """Confidence < 0.6 should return 'red'."""
        assert get_confidence_color(0.59) == "red"
        assert get_confidence_color(0.5) == "red"
        assert get_confidence_color(0.0) == "red"
        assert get_confidence_color(0.3) == "red"


class TestFormatConfidence:
    """Tests for format_confidence function."""
    
    def test_format_100_percent(self):
        """Should format 1.0 as '100.0%'."""
        assert format_confidence(1.0) == "100.0%"
    
    def test_format_50_percent(self):
        """Should format 0.5 as '50.0%'."""
        assert format_confidence(0.5) == "50.0%"
    
    def test_format_0_percent(self):
        """Should format 0.0 as '0.0%'."""
        assert format_confidence(0.0) == "0.0%"
    
    def test_format_decimal(self):
        """Should format decimal values correctly."""
        assert format_confidence(0.92) == "92.0%"
        assert format_confidence(0.333) == "33.3%"
    
    def test_format_returns_string(self):
        """Should return a string."""
        result = format_confidence(0.75)
        assert isinstance(result, str)


class TestHighlightDrugNames:
    """Tests for highlight_drug_names function."""
    
    def test_highlight_single_drug(self):
        """Should wrap single drug name in **.**"""
        text = "Take Aspirin for pain"
        result = highlight_drug_names(text, ["Aspirin"])
        assert "**Aspirin**" in result
    
    def test_highlight_multiple_drugs(self):
        """Should wrap multiple drug names."""
        text = "Aspirin and Tylenol"
        result = highlight_drug_names(text, ["Aspirin", "Tylenol"])
        assert "**Aspirin**" in result
        assert "**Tylenol**" in result
    
    def test_highlight_case_insensitive(self):
        """Should highlight regardless of case."""
        text = "Take ASPIRIN for pain"
        result = highlight_drug_names(text, ["aspirin"])
        assert "**ASPIRIN**" in result
    
    def test_no_drug_list_returns_original(self):
        """Should return original text if no drug list."""
        text = "Take medicine"
        result = highlight_drug_names(text)
        assert result == text
    
    def test_empty_drug_list_returns_original(self):
        """Should return original text if drug list is empty."""
        text = "Take medicine"
        result = highlight_drug_names(text, [])
        assert result == text
    
    def test_no_match_returns_original(self):
        """Should return original text if no drug matches."""
        text = "Take medicine"
        result = highlight_drug_names(text, ["Aspirin"])
        assert result == text


class TestSaveScanToHistory:
    """Tests for save_scan_to_history function."""
    
    def test_adds_new_entry(self, empty_history, sample_result, image_hash, max_history_size):
        """Should add a new entry to history."""
        result = save_scan_to_history(empty_history, sample_result, image_hash, max_history_size)
        assert len(result) == 1
        assert result[0]["image_hash"] == image_hash
    
    def test_truncates_to_max_history(self, sample_history, sample_result, image_hash):
        """Should truncate history to max_history size."""
        max_size = 5
        result = save_scan_to_history(sample_history * 3, sample_result, image_hash, max_size)
        assert len(result) <= max_size
    
    def test_entry_contains_required_fields(self, empty_history, sample_result, image_hash, max_history_size):
        """Should contain all required fields."""
        result = save_scan_to_history(empty_history, sample_result, image_hash, max_history_size)
        entry = result[0]
        assert "time" in entry
        assert "image_hash" in entry
        assert "preview" in entry
        assert "confidence" in entry
        assert "line_count" in entry
    
    def test_preview_truncated(self, empty_history, sample_result, image_hash, max_history_size):
        """Preview should be truncated to 50 chars."""
        result = save_scan_to_history(empty_history, sample_result, image_hash, max_history_size)
        assert len(result[0]["preview"]) <= 50


class TestExportToJson:
    """Tests for export_to_json function."""
    
    def test_returns_bytes(self, sample_result):
        """Should return bytes."""
        result = export_to_json(sample_result)
        assert isinstance(result, bytes)
    
    def test_valid_json(self, sample_result):
        """Should produce valid JSON when decoded."""
        result = export_to_json(sample_result)
        decoded = json.loads(result.decode("utf-8"))
        assert "lines" in decoded
        assert len(decoded["lines"]) == sample_result["total_lines"]
    
    def test_round_trip(self, sample_result):
        """JSON should be decodable and match original."""
        result = export_to_json(sample_result)
        decoded = json.loads(result.decode("utf-8"))
        assert decoded["total_lines"] == sample_result["total_lines"]
        assert decoded["avg_confidence"] == sample_result["avg_confidence"]


class TestExportToTxt:
    """Tests for export_to_txt function."""
    
    def test_returns_bytes(self, sample_result):
        """Should return bytes."""
        result = export_to_txt(sample_result)
        assert isinstance(result, bytes)
    
    def test_contains_header(self, sample_result):
        """Should contain the result header."""
        result = export_to_txt(sample_result)
        decoded = result.decode("utf-8")
        assert "PRESCRIPTION OCR RESULT" in decoded
    
    def test_contains_lines(self, sample_result):
        """Should contain extracted lines."""
        result = export_to_txt(sample_result)
        decoded = result.decode("utf-8")
        assert "Line 1:" in decoded
        assert sample_result["lines"][0]["text"] in decoded
    
    def test_contains_stats(self, sample_result):
        """Should contain summary statistics."""
        result = export_to_txt(sample_result)
        decoded = result.decode("utf-8")
        assert "Total Lines:" in decoded
        assert "Average Confidence:" in decoded
    
    def test_flagged_warning(self, sample_result):
        """Should include warning for flagged lines."""
        result = export_to_txt(sample_result)
        decoded = result.decode("utf-8")
        assert "Low confidence" in decoded


class TestExportToCsv:
    """Tests for export_to_csv function."""
    
    def test_returns_bytes(self, sample_result):
        """Should return bytes."""
        result = export_to_csv(sample_result)
        assert isinstance(result, bytes)
    
    def test_contains_header_row(self, sample_result):
        """Should contain CSV header row."""
        result = export_to_csv(sample_result)
        decoded = result.decode("utf-8")
        assert "Line ID" in decoded
        assert "Text" in decoded
        assert "Confidence" in decoded
    
    def test_contains_data_rows(self, sample_result):
        """Should contain data for each line."""
        result = export_to_csv(sample_result)
        decoded = result.decode("utf-8")
        lines = decoded.strip().split("\n")
        # Header + 4 data rows
        assert len(lines) == sample_result["total_lines"] + 1
    
    def test_flagged_column(self, sample_result):
        """Should have Yes/No for flagged column."""
        result = export_to_csv(sample_result)
        decoded = result.decode("utf-8")
        assert "Yes" in decoded or "No" in decoded