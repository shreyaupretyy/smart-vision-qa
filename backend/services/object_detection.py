from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Object detection service using YOLOv8"""
    
    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        Initialize object detector
        
        Args:
            model_name: YOLO model to use (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_name)
            logger.info(f"Loaded YOLO model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_objects(
        self, 
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ) -> List[Dict]:
        """
        Detect objects in a frame
        
        Args:
            frame: Input frame as numpy array
            confidence_threshold: Minimum confidence score
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        results = self.model(frame, conf=confidence_threshold, iou=iou_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        
        return detections
    
    def detect_in_video(
        self,
        video_path: str,
        confidence_threshold: float = 0.5,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        sample_rate: int = 1
    ) -> List[Dict]:
        """
        Detect objects throughout a video
        
        Args:
            video_path: Path to video file
            confidence_threshold: Minimum confidence score
            start_frame: Starting frame number
            end_frame: Ending frame number (None for all frames)
            sample_rate: Process every Nth frame
            
        Returns:
            List of detections with frame numbers and timestamps
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if end_frame is None:
            end_frame = total_frames
        
        all_detections = []
        
        for frame_num in range(start_frame, end_frame, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            detections = self.detect_objects(frame, confidence_threshold)
            
            timestamp = frame_num / fps
            for detection in detections:
                detection["frame_number"] = frame_num
                detection["timestamp"] = timestamp
                all_detections.append(detection)
        
        cap.release()
        logger.info(f"Detected {len(all_detections)} objects in video")
        return all_detections
    
    def track_objects(
        self,
        video_path: str,
        confidence_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Track objects across video frames
        
        Args:
            video_path: Path to video file
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of tracked objects with trajectories
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        results = self.model.track(
            video_path, 
            conf=confidence_threshold,
            persist=True,
            verbose=False
        )
        
        tracks = []
        for i, result in enumerate(results):
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes
                for j in range(len(boxes)):
                    track = {
                        "frame_number": i,
                        "track_id": int(boxes.id[j]),
                        "class_name": result.names[int(boxes.cls[j])],
                        "confidence": float(boxes.conf[j]),
                        "bbox": boxes.xyxy[j].tolist(),
                    }
                    tracks.append(track)
        
        logger.info(f"Tracked {len(tracks)} object instances")
        return tracks
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        draw_labels: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            draw_labels: Whether to draw labels
            
        Returns:
            Frame with drawn detections
        """
        output_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["bbox"])
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if draw_labels:
                label = f"{detection['class_name']} {detection['confidence']:.2f}"
                
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    output_frame,
                    (x1, y1 - label_height - 10),
                    (x1 + label_width, y1),
                    (0, 255, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    output_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
        
        return output_frame
    
    def get_object_counts(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Count detected objects by class
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with class names and counts
        """
        counts = {}
        for detection in detections:
            class_name = detection["class_name"]
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts
    
    def filter_detections_by_class(
        self,
        detections: List[Dict],
        class_names: List[str]
    ) -> List[Dict]:
        """
        Filter detections by class names
        
        Args:
            detections: List of detections
            class_names: List of class names to keep
            
        Returns:
            Filtered detections
        """
        return [d for d in detections if d["class_name"] in class_names]
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict:
        """
        Get summary statistics of detections
        
        Args:
            detections: List of detections
            
        Returns:
            Summary dictionary
        """
        if not detections:
            return {
                "total_detections": 0,
                "unique_classes": 0,
                "class_counts": {},
                "avg_confidence": 0.0
            }
        
        class_counts = self.get_object_counts(detections)
        confidences = [d["confidence"] for d in detections]
        
        return {
            "total_detections": len(detections),
            "unique_classes": len(class_counts),
            "class_counts": class_counts,
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences)
        }
