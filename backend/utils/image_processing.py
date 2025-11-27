"""
Image processing utilities for frames and thumbnails.
"""
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

def resize_image(
    image_path: str,
    output_path: str,
    max_width: int = 1280,
    max_height: int = 720,
    quality: int = 85
) -> bool:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image_path: Path to input image
        output_path: Path to save resized image
        max_width: Maximum width
        max_height: Maximum height
        quality: JPEG quality (1-100)
        
    Returns:
        True if successful
    """
    try:
        img = Image.open(image_path)
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        img.save(output_path, "JPEG", quality=quality)
        return True
    except Exception as e:
        print(f"Error resizing image: {e}")
        return False

def create_thumbnail(
    image_path: str,
    output_path: str,
    size: Tuple[int, int] = (320, 180)
) -> bool:
    """
    Create thumbnail from image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save thumbnail
        size: Thumbnail size (width, height)
        
    Returns:
        True if successful
    """
    try:
        img = Image.open(image_path)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        img.save(output_path, "JPEG", quality=70)
        return True
    except Exception as e:
        print(f"Error creating thumbnail: {e}")
        return False

def extract_video_frame(
    video_path: str,
    frame_number: int,
    output_path: str
) -> bool:
    """
    Extract specific frame from video.
    
    Args:
        video_path: Path to video file
        frame_number: Frame number to extract
        output_path: Path to save frame
        
    Returns:
        True if successful
    """
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            cv2.imwrite(output_path, frame)
            return True
        return False
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return False

def apply_blur(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    blur_strength: int = 51
) -> np.ndarray:
    """
    Apply Gaussian blur to specific region.
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        blur_strength: Blur kernel size (must be odd)
        
    Returns:
        Image with blurred region
    """
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
    image[y1:y2, x1:x2] = blurred
    return image

def apply_pixelate(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pixel_size: int = 20
) -> np.ndarray:
    """
    Apply pixelation to specific region.
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        pixel_size: Size of pixelation blocks
        
    Returns:
        Image with pixelated region
    """
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    
    # Downscale
    h, w = roi.shape[:2]
    temp = cv2.resize(roi, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    
    # Upscale back
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    
    image[y1:y2, x1:x2] = pixelated
    return image

def apply_black_box(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Apply black box to specific region.
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Image with black box
    """
    x1, y1, x2, y2 = bbox
    image[y1:y2, x1:x2] = 0
    return image
