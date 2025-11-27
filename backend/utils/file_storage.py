"""
File storage utilities for managing uploaded videos and processed files.
"""
import os
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime
from backend.core.config import settings

def ensure_upload_dirs():
    """Create upload directories if they don't exist."""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.FRAMES_DIR, exist_ok=True)
    os.makedirs(settings.PROCESSED_DIR, exist_ok=True)

def get_video_path(video_id: str) -> Path:
    """
    Get path to uploaded video file.
    
    Args:
        video_id: Video identifier
        
    Returns:
        Path to video file
    """
    return Path(settings.UPLOAD_DIR) / f"{video_id}.mp4"

def get_frames_dir(video_id: str) -> Path:
    """
    Get directory for video frames.
    
    Args:
        video_id: Video identifier
        
    Returns:
        Path to frames directory
    """
    frames_dir = Path(settings.FRAMES_DIR) / video_id
    os.makedirs(frames_dir, exist_ok=True)
    return frames_dir

def get_processed_video_path(video_id: str, suffix: str = "processed") -> Path:
    """
    Get path for processed video output.
    
    Args:
        video_id: Video identifier
        suffix: Filename suffix
        
    Returns:
        Path to processed video
    """
    return Path(settings.PROCESSED_DIR) / f"{video_id}_{suffix}.mp4"

def delete_video_files(video_id: str) -> bool:
    """
    Delete all files associated with a video.
    
    Args:
        video_id: Video identifier
        
    Returns:
        True if successful
    """
    try:
        # Delete original video
        video_path = get_video_path(video_id)
        if video_path.exists():
            os.remove(video_path)
        
        # Delete frames directory
        frames_dir = get_frames_dir(video_id)
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        
        # Delete processed videos
        processed_dir = Path(settings.PROCESSED_DIR)
        for file in processed_dir.glob(f"{video_id}_*.mp4"):
            os.remove(file)
        
        return True
    except Exception as e:
        print(f"Error deleting video files: {e}")
        return False

def get_file_size(file_path: Path) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path) if file_path.exists() else 0

def cleanup_old_files(days: int = 7):
    """
    Clean up files older than specified days.
    
    Args:
        days: Number of days to keep files
    """
    cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
    
    for directory in [settings.UPLOAD_DIR, settings.FRAMES_DIR, settings.PROCESSED_DIR]:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        os.remove(file_path)
                        print(f"Deleted old file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
