"""
Video validation utilities.
"""
import os
from pathlib import Path
from typing import Tuple, Optional
import magic
from backend.core.constants import (
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_VIDEO_SIZE,
    MAX_VIDEO_DURATION
)

def validate_video_extension(filename: str) -> bool:
    """
    Validate video file extension.
    
    Args:
        filename: Name of the video file
        
    Returns:
        True if extension is allowed
    """
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_VIDEO_EXTENSIONS

def validate_video_size(file_size: int) -> bool:
    """
    Validate video file size.
    
    Args:
        file_size: Size of file in bytes
        
    Returns:
        True if size is within limits
    """
    return 0 < file_size <= MAX_VIDEO_SIZE

def validate_video_mime_type(file_path: str) -> bool:
    """
    Validate video MIME type.
    
    Args:
        file_path: Path to video file
        
    Returns:
        True if MIME type is valid video format
    """
    try:
        mime = magic.Magic(mime=True)
        file_mime = mime.from_file(file_path)
        return file_mime.startswith('video/')
    except:
        # Fallback if python-magic is not available
        return validate_video_extension(file_path)

def validate_video_file(
    filename: str,
    file_size: int,
    file_path: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Comprehensive video file validation.
    
    Args:
        filename: Name of the video file
        file_size: Size of file in bytes
        file_path: Optional path to file for MIME type check
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check extension
    if not validate_video_extension(filename):
        return False, f"Invalid file extension. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
    
    # Check size
    if not validate_video_size(file_size):
        max_mb = MAX_VIDEO_SIZE / (1024 * 1024)
        return False, f"File size exceeds maximum limit of {max_mb}MB"
    
    # Check MIME type if path provided
    if file_path and not validate_video_mime_type(file_path):
        return False, "Invalid video file format"
    
    return True, None

def get_safe_filename(filename: str) -> str:
    """
    Generate safe filename by removing special characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Keep only alphanumeric, dash, underscore, and dot
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in ['-', '_', '.']:
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    
    return ''.join(safe_chars)
