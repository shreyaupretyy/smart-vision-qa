"""
Pytest configuration and fixtures for SmartVisionQA tests.
"""
import pytest
import os
from pathlib import Path

# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
SAMPLE_VIDEO_PATH = TEST_DATA_DIR / "sample_video.mp4"

@pytest.fixture
def test_data_dir():
    """Returns the test data directory path."""
    return TEST_DATA_DIR

@pytest.fixture
def sample_video_path():
    """Returns path to sample test video."""
    return SAMPLE_VIDEO_PATH

@pytest.fixture
def mock_video_metadata():
    """Mock video metadata for testing."""
    return {
        "filename": "test_video.mp4",
        "duration": 30.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "frame_count": 900,
        "size": 1024000
    }

@pytest.fixture
def mock_detection_results():
    """Mock object detection results."""
    return [
        {
            "frame_number": 1,
            "timestamp": 0.0,
            "detections": [
                {"class": "person", "confidence": 0.95, "bbox": [100, 100, 200, 300]},
                {"class": "car", "confidence": 0.88, "bbox": [400, 300, 600, 500]}
            ]
        }
    ]

@pytest.fixture
def mock_transcription():
    """Mock transcription results."""
    return {
        "text": "This is a test transcription.",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "This is a test"},
            {"start": 2.5, "end": 5.0, "text": "transcription."}
        ],
        "language": "en"
    }
