"""
Utility functions for SmartVisionQA
"""
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_file_size(bytes: int) -> str:
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def validate_video_format(filename: str) -> bool:
    """Validate if file is a supported video format"""
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)


def merge_overlapping_detections(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """Merge overlapping bounding boxes"""
    # Simple implementation - can be improved
    if not detections:
        return []
    
    # Sort by confidence
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    merged = []
    
    for det in sorted_dets:
        overlap = False
        for m in merged:
            if calculate_iou(det['bbox'], m['bbox']) > iou_threshold:
                overlap = True
                break
        if not overlap:
            merged.append(det)
    
    return merged


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    if x_max < x_min or y_max < y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0
