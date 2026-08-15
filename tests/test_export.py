"""
Tests for export functions

Tests JSON, TXT, and CSV export functionality.
"""

import os
import sys
import json
import pytest
import csv
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frontend.utils import export_to_json, export_to_txt, export_to_csv


class TestExportToJson:
    """Comprehensive tests for export_to_json function."""
    
    def test_returns_bytes(self, sample_result):
        """Should return bytes."""
        result = export_to_json(sample_result)
        assert isinstance(result, bytes)
    
    def test_valid_json_structure(self, sample_result):
        """Should produce valid JSON."""
        result = export_to_json(sample_result)
        decoded = json.loads(result.decode("utf-8"))
        assert isinstance(decoded, dict)
    
    def test_contains_all_required_fields(self, sample_result):
        """Should contain all result fields."""
        result = export_to_json(sample_result)
        decoded = json.loads(result.decode("utf-8"))
        assert "lines" in decoded
        assert "total_lines" in decoded
        assert "avg_confidence" in decoded
        assert "abbreviations_found" in decoded
        assert "processing_time" in decoded
        assert "model" in decoded
        assert "raw_text" in decoded
    
    def test_lines_array_structure(self, sample_result):
        """Lines should have correct structure."""
        result = export_to_json(sample_result)
        decoded = json.loads(result.decode("utf-8"))
        assert isinstance(decoded["lines"], list)
        assert len(decoded["lines"]) == sample_result["total_lines"]
    
    def test_line_structure(self, sample_result):
        """Each line should have required fields."""
        result = export_to_json(sample_result)
        decoded = json.loads(result.decode("utf-8"))
        for line in decoded["lines"]:
            assert "line_id" in line
            assert "text" in line
            assert "expanded" in line
            assert "confidence" in line
            assert "flagged" in line
            assert "abbreviations_in_line" in line
    
    def test_confidence_values_preserved(self, sample_result):
        """Confidence values should be preserved."""
        result = export_to_json(sample_result)
        decoded = json.loads(result.decode("utf-8"))
        for i, line in enumerate(decoded["lines"]):
            assert line["confidence"] == sample_result["lines"][i]["confidence"]
    
    def test_empty_result(self, empty_result):
        """Should handle empty result."""
        result = export_to_json(empty_result)
        decoded = json.loads(result.decode("utf-8"))
        assert decoded["total_lines"] == 0
        assert decoded["lines"] == []
    
    def test_single_line_result(self, single_line_result):
        """Should handle single line result."""
        result = export_to_json(single_line_result)
        decoded = json.loads(result.decode("utf-8"))
        assert decoded["total_lines"] == 1
        assert len(decoded["lines"]) == 1
    
    def test_indentation(self, sample_result):
        """Should have proper indentation."""
        result = export_to_json(sample_result)
        decoded = result.decode("utf-8")
        # Should contain newlines (indicating indentation)
        assert "\n" in decoded


class TestExportToTxt:
    """Comprehensive tests for export_to_txt function."""
    
    def test_returns_bytes(self, sample_result):
        """Should return bytes."""
        result = export_to_txt(sample_result)
        assert isinstance(result, bytes)
    
    def test_contains_header(self, sample_result):
        """Should contain header section."""
        result = export_to_txt(sample_result)
        decoded = result.decode("utf-8")
        assert "PRESCRIPTION OCR RESULT" in decoded
    
    def test_contains_divider_lines(self, sample_result):
        """Should contain divider lines."""
        result = export_to_txt(sample_result)
        decoded = result.decode("utf-8")
        assert "=" * 50 in decoded
    
    def test_contains_all_lines(self, sample_result):
        """Should contain all extracted lines."""
        result = export_to_txt(sample_result)
        decoded = result.decode("utf-8")
        for line in sample_result["lines"]:
            assert f"Line {line['line_id']}:" in decoded
            assert line["text"] in decoded
    
    def test_contains_expanded_text(self, sample_result):
        """Should contain expanded text for abbreviated lines."""
        result = export_to_txt(sample_result)
        decoded = result.decode("utf-8")
        assert "→" in decoded
    
    def test_contains_confidence(self, sample_result):
        """Should contain confidence scores."""
        result = export_to_txt(sample_result)
        decoded = result.decode("utf-8")
        assert "Confidence:" in decoded
        assert "%" in decoded
    
    def test_contains_flagged_warning(self, sample_result):
        """Should contain warning for flagged lines."""
        result = export_to_txt(sample_result)
        decoded = result.decode("utf-8")
        assert "Low confidence" in decoded
    
    def test_contains_summary(self, sample_result):
        """Should contain summary statistics."""
        result = export_to_txt(sample_result)
        decoded = result.decode("utf-8")
        assert "Total Lines:" in decoded
        assert "Average Confidence:" in decoded
        assert "Abbreviations Found:" in decoded
        assert "Model:" in decoded
    
    def test_empty_result(self, empty_result):
        """Should handle empty result gracefully."""
        result = export_to_txt(empty_result)
        decoded = result.decode("utf-8")
        assert "PRESCRIPTION OCR RESULT" in decoded
    
    def test_single_line_result(self, single_line_result):
        """Should handle single line result."""
        result = export_to_txt(single_line_result)
        decoded = result.decode("utf-8")
        assert "Line 1:" in decoded


class TestExportToCsv:
    """Comprehensive tests for export_to_csv function."""
    
    def test_returns_bytes(self, sample_result):
        """Should return bytes."""
        result = export_to_csv(sample_result)
        assert isinstance(result, bytes)
    
    def test_valid_csv_structure(self, sample_result):
        """Should produce valid CSV."""
        result = export_to_csv(sample_result)
        decoded = result.decode("utf-8")
        lines = decoded.strip().split("\n")
        assert len(lines) >= 2  # Header + at least one data row
    
    def test_header_columns(self, sample_result):
        """Should have correct header columns."""
        result = export_to_csv(sample_result)
        decoded = result.decode("utf-8")
        reader = csv.reader(StringIO(decoded))
        header = next(reader)
        expected = ["Line ID", "Text", "Expanded", "Confidence", "Flagged", "Abbreviations"]
        assert header == expected
    
    def test_data_rows_count(self, sample_result):
        """Should have correct number of data rows."""
        result = export_to_csv(sample_result)
        decoded = result.decode("utf-8")
        reader = csv.reader(StringIO(decoded))
        next(reader)  # Skip header
        rows = list(reader)
        assert len(rows) == sample_result["total_lines"]
    
    def test_confidence_format(self, sample_result):
        """Confidence should be in percentage format."""
        result = export_to_csv(sample_result)
        decoded = result.decode("utf-8")
        reader = csv.reader(StringIO(decoded))
        next(reader)  # Skip header
        for row in reader:
            confidence = row[3]  # Confidence column
            assert "%" in confidence
    
    def test_flagged_values(self, sample_result):
        """Flagged column should be Yes/No."""
        result = export_to_csv(sample_result)
        decoded = result.decode("utf-8")
        reader = csv.reader(StringIO(decoded))
        next(reader)  # Skip header
        for row in reader:
            flagged = row[4]  # Flagged column
            assert flagged in ["Yes", "No"]
    
    def test_abbreviations_count(self, sample_result):
        """Abbreviations column should be integer."""
        result = export_to_csv(sample_result)
        decoded = result.decode("utf-8")
        reader = csv.reader(StringIO(decoded))
        next(reader)  # Skip header
        for row in reader:
            abbrev_count = row[5]  # Abbreviations column
            assert int(abbrev_count) >= 0
    
    def test_line_id_column(self, sample_result):
        """Line ID column should be present and correct."""
        result = export_to_csv(sample_result)
        decoded = result.decode("utf-8")
        reader = csv.reader(StringIO(decoded))
        next(reader)  # Skip header
        for i, row in enumerate(reader):
            expected_id = i + 1
            assert int(row[0]) == expected_id
    
    def test_empty_result(self, empty_result):
        """Should handle empty result."""
        result = export_to_csv(empty_result)
        decoded = result.decode("utf-8")
        reader = csv.reader(StringIO(decoded))
        header = next(reader)
        rows = list(reader)
        assert len(rows) == 0
    
    def test_single_line_result(self, single_line_result):
        """Should handle single line result."""
        result = export_to_csv(single_line_result)
        decoded = result.decode("utf-8")
        reader = csv.reader(StringIO(decoded))
        next(reader)  # Skip header
        rows = list(reader)
        assert len(rows) == 1