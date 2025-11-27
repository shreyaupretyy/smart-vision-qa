"""
Unit tests for video processor service
"""
import pytest
import numpy as np
from pathlib import Path
from backend.services.video_processor import VideoProcessor


@pytest.fixture
def video_processor():
    return VideoProcessor(upload_dir="./test_uploads", temp_dir="./test_temp")


def test_video_processor_initialization(video_processor):
    """Test video processor initialization"""
    assert video_processor.upload_dir.exists()
    assert video_processor.temp_dir.exists()


def test_resize_frame(video_processor):
    """Test frame resizing"""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    resized = video_processor.resize_frame(frame, 320, 240)
    assert resized.shape == (240, 320, 3)


def test_chunk_video_timing():
    """Test video chunking logic"""
    # This would require a test video file
    pass
