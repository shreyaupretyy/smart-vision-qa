import cv2
import os
import uuid
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Video processing service for frame extraction and manipulation"""
    
    def __init__(self, upload_dir: str = "./uploads", temp_dir: str = "./temp"):
        self.upload_dir = Path(upload_dir)
        self.temp_dir = Path(temp_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def save_uploaded_video(self, file_content: bytes, filename: str) -> Tuple[str, str]:
        """
        Save uploaded video file
        
        Args:
            file_content: Video file content
            filename: Original filename
            
        Returns:
            Tuple of (video_id, file_path)
        """
        video_id = str(uuid.uuid4())
        ext = Path(filename).suffix
        file_path = self.upload_dir / f"{video_id}{ext}"
        
        with open(file_path, "wb") as f:
            f.write(file_content)
            
        logger.info(f"Saved video {video_id} to {file_path}")
        return video_id, str(file_path)
    
    def get_video_metadata(self, video_path: str) -> dict:
        """
        Extract video metadata
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        metadata = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        }
        
        cap.release()
        logger.info(f"Extracted metadata from {video_path}: {metadata}")
        return metadata
    
    def extract_frames(
        self, 
        video_path: str, 
        sample_rate: int = 1,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            sample_rate: Extract 1 frame every N seconds
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            
        Returns:
            List of tuples (frame_number, timestamp, frame_array)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate start and end frames
        start_frame = int(start_time * fps) if start_time else 0
        end_frame = int(end_time * fps) if end_time else total_frames
        
        frames = []
        frame_interval = int(fps * sample_rate)
        
        for frame_num in range(start_frame, end_frame, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            timestamp = frame_num / fps
            frames.append((frame_num, timestamp, frame))
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def get_frame_at_time(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """
        Get a specific frame at timestamp
        
        Args:
            video_path: Path to video file
            timestamp: Time in seconds
            
        Returns:
            Frame as numpy array or None
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        cap.release()
        
        return frame if ret else None
    
    def save_frame(self, frame: np.ndarray, output_path: str) -> str:
        """
        Save a frame as image
        
        Args:
            frame: Frame as numpy array
            output_path: Output file path
            
        Returns:
            Path to saved image
        """
        cv2.imwrite(output_path, frame)
        logger.info(f"Saved frame to {output_path}")
        return output_path
    
    def create_video_from_frames(
        self, 
        frames: List[np.ndarray], 
        output_path: str,
        fps: float = 30.0
    ) -> str:
        """
        Create video from list of frames
        
        Args:
            frames: List of frames as numpy arrays
            output_path: Output video path
            fps: Frames per second
            
        Returns:
            Path to output video
        """
        if not frames:
            raise ValueError("No frames provided")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Created video at {output_path}")
        return output_path
    
    def resize_frame(
        self, 
        frame: np.ndarray, 
        target_width: int = 640,
        target_height: int = 480
    ) -> np.ndarray:
        """
        Resize frame to target dimensions
        
        Args:
            frame: Input frame
            target_width: Target width
            target_height: Target height
            
        Returns:
            Resized frame
        """
        return cv2.resize(frame, (target_width, target_height))
    
    def get_video_path(self, video_id: str) -> Optional[str]:
        """
        Get path to video file by ID
        
        Args:
            video_id: Video ID
            
        Returns:
            Path to video file or None
        """
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            path = self.upload_dir / f"{video_id}{ext}"
            if path.exists():
                return str(path)
        return None
    
    def delete_video(self, video_id: str) -> bool:
        """
        Delete video file
        
        Args:
            video_id: Video ID
            
        Returns:
            True if deleted, False otherwise
        """
        video_path = self.get_video_path(video_id)
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
            logger.info(f"Deleted video {video_id}")
            return True
        return False
    
    def chunk_video(
        self, 
        video_path: str, 
        chunk_duration: int = 10
    ) -> List[Tuple[float, float]]:
        """
        Divide video into time chunks
        
        Args:
            video_path: Path to video file
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            List of (start_time, end_time) tuples
        """
        metadata = self.get_video_metadata(video_path)
        duration = metadata["duration"]
        
        chunks = []
        current_time = 0
        
        while current_time < duration:
            end_time = min(current_time + chunk_duration, duration)
            chunks.append((current_time, end_time))
            current_time = end_time
        
        logger.info(f"Divided video into {len(chunks)} chunks")
        return chunks
