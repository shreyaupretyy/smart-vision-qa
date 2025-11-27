"""
Unit tests for object detection service
"""
import pytest
import numpy as np
from backend.services.object_detection import ObjectDetector


@pytest.fixture
def detector():
    return ObjectDetector("yolov8n.pt")


def test_object_counts(detector):
    """Test object counting functionality"""
    detections = [
        {"class_name": "person", "confidence": 0.9},
        {"class_name": "person", "confidence": 0.85},
        {"class_name": "car", "confidence": 0.8},
    ]
    counts = detector.get_object_counts(detections)
    assert counts["person"] == 2
    assert counts["car"] == 1


def test_filter_detections(detector):
    """Test detection filtering"""
    detections = [
        {"class_name": "person", "confidence": 0.9},
        {"class_name": "car", "confidence": 0.8},
        {"class_name": "dog", "confidence": 0.75},
    ]
    filtered = detector.filter_detections_by_class(detections, ["person", "dog"])
    assert len(filtered) == 2
