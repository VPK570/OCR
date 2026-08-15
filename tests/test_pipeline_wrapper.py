"""
Tests for frontend/pipeline_wrapper.py

Tests the pipeline wrapper that bridges Streamlit to the OCR pipeline.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frontend.pipeline_wrapper import run_ocr_pipeline


class TestRunOcrPipeline:
    """Tests for run_ocr_pipeline function."""
    
    def test_returns_dict(self, sample_image, mock_pipeline_run):
        """Should return a dictionary."""
        result = run_ocr_pipeline(sample_image)
        assert isinstance(result, dict)
    
    def test_has_required_keys(self, sample_image, mock_pipeline_run):
        """Should have all required keys."""
        result = run_ocr_pipeline(sample_image)
        required_keys = [
            "lines", "total_lines", "avg_confidence", 
            "abbreviations_found", "processing_time", 
            "model", "raw_text"
        ]
        for key in required_keys:
            assert key in result
    
    def test_lines_structure(self, sample_image, mock_pipeline_run):
        """Lines should have correct structure."""
        result = run_ocr_pipeline(sample_image)
        if result["lines"]:
            line = result["lines"][0]
            assert "line_id" in line
            assert "text" in line
            assert "expanded" in line
            assert "confidence" in line
            assert "flagged" in line
            assert "abbreviations_in_line" in line
    
    def test_confidence_range(self, sample_image, mock_pipeline_run):
        """Confidence should be in 0.0-1.0 range."""
        result = run_ocr_pipeline(sample_image)
        for line in result["lines"]:
            assert 0.0 <= line["confidence"] <= 1.0
    
    def test_model_set(self, sample_image, mock_pipeline_run):
        """Should set model from recognizer parameter."""
        result = run_ocr_pipeline(sample_image, recognizer="TrOCR")
        assert result["model"] == "TrOCR"
        
        result = run_ocr_pipeline(sample_image, recognizer="HTRVT")
        assert result["model"] == "HTRVT"
    
    def test_abbreviation_expansion_enabled(self, sample_image, mock_pipeline_run):
        """Should expand abbreviations when enabled."""
        result = run_ocr_pipeline(sample_image, use_abbreviations=True)
        for line in result["lines"]:
            # Expanded should be present
            assert "expanded" in line
    
    def test_abbreviation_expansion_disabled(self, sample_image, mock_pipeline_run):
        """Should not expand abbreviations when disabled."""
        result = run_ocr_pipeline(sample_image, use_abbreviations=False)
        for line in result["lines"]:
            # Text and expanded should be same
            assert line["text"] == line["expanded"]
    
    def test_confidence_threshold_flagging(self, sample_image, mock_pipeline_run_low_confidence):
        """Lines below threshold should be flagged."""
        result = run_ocr_pipeline(sample_image, confidence_threshold=0.5)
        for line in result["lines"]:
            if line["confidence"] < 0.5:
                assert line["flagged"] is True


class TestRunOcrPipelineEdgeCases:
    """Edge case tests for run_ocr_pipeline function."""
    
    def test_empty_result(self, sample_image, mock_pipeline_run_empty):
        """Should handle empty pipeline result."""
        result = run_ocr_pipeline(sample_image)
        assert result["total_lines"] == 0
        assert result["lines"] == []
        assert result["avg_confidence"] == 0.0
    
    def test_empty_image(self, mock_pipeline_run_empty):
        """Should handle empty (all black) image."""
        empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = run_ocr_pipeline(empty_img)
        assert isinstance(result, dict)
    
    def test_different_image_sizes(self, mock_pipeline_run):
        """Should handle different image sizes."""
        # Small image
        small_img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = run_ocr_pipeline(small_img)
        assert isinstance(result, dict)
        
        # Large image
        large_img = np.zeros((500, 500, 3), dtype=np.uint8)
        result = run_ocr_pipeline(large_img)
        assert isinstance(result, dict)
    
    def test_grayscale_image(self, mock_pipeline_run):
        """Should handle grayscale image (converted to BGR)."""
        gray_img = np.zeros((100, 200), dtype=np.uint8)
        result = run_ocr_pipeline(gray_img)
        assert isinstance(result, dict)


class TestRunOcrPipelineTempFileCleanup:
    """Tests for temporary file cleanup."""
    
    def test_temp_file_created_and_deleted(self, sample_image, mock_pipeline_run):
        """Should clean up temp files after processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_temp_dir = tempfile.tempdir
            tempfile.tempdir = tmpdir
            
            try:
                run_ocr_pipeline(sample_image)
                
                # Check for any temp files created
                temp_files = [f for f in os.listdir(tmpdir) if f.startswith("ocr_temp_")]
                assert len(temp_files) == 0
            finally:
                tempfile.tempdir = original_temp_dir
    
    def test_temp_file_cleanup_on_exception(self, sample_image, mock_pipeline_run):
        """Should clean up temp files even on exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_temp_dir = tempfile.tempdir
            tempfile.tempdir = tmpdir
            
            try:
                try:
                    run_ocr_pipeline(sample_image)
                except Exception:
                    pass
                
                temp_files = [f for f in os.listdir(tmpdir) if f.startswith("ocr_temp_")]
                assert len(temp_files) == 0
            finally:
                tempfile.tempdir = original_temp_dir


class TestRunOcrPipelineAbbreviationCount:
    """Tests for abbreviation counting."""
    
    def test_counts_abbreviations(self, sample_image, mock_pipeline_run):
        """Should count abbreviations correctly."""
        result = run_ocr_pipeline(sample_image)
        assert result["abbreviations_found"] >= 0
        assert isinstance(result["abbreviations_found"], int)
    
    def test_abbreviation_count_in_lines(self, sample_image, mock_pipeline_run):
        """Lines should have abbreviation counts."""
        result = run_ocr_pipeline(sample_image)
        if result["lines"]:
            for line in result["lines"]:
                assert "abbreviations_in_line" in line
                assert isinstance(line["abbreviations_in_line"], int)
                assert line["abbreviations_in_line"] >= 0