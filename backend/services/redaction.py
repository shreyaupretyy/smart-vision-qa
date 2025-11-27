import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoRedactor:
    """Video redaction service for privacy protection"""
    
    def __init__(self):
        """Initialize video redactor"""
        # Load face detection model
        self.face_cascade = None
        self._load_face_detector()
    
    def _load_face_detector(self):
        """Load Haar cascade for face detection"""
        try:
            # Try to load from OpenCV data
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Face detector loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load face detector: {e}")
    
    def detect_faces(
        self, 
        frame: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame
        
        Args:
            frame: Input frame
            scale_factor: Scale factor for detection
            min_neighbors: Minimum neighbors for detection
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        if self.face_cascade is None:
            logger.warning("Face detector not loaded")
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        
        return faces.tolist() if len(faces) > 0 else []
    
    def blur_region(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        blur_intensity: int = 50
    ) -> np.ndarray:
        """
        Blur a specific region in frame
        
        Args:
            frame: Input frame
            x, y, w, h: Region coordinates
            blur_intensity: Blur kernel size (must be odd)
            
        Returns:
            Frame with blurred region
        """
        output = frame.copy()
        
        # Ensure blur intensity is odd
        if blur_intensity % 2 == 0:
            blur_intensity += 1
        
        # Extract region
        region = frame[y:y+h, x:x+w]
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(region, (blur_intensity, blur_intensity), 0)
        
        # Replace region
        output[y:y+h, x:x+w] = blurred
        
        return output
    
    def pixelate_region(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        pixel_size: int = 20
    ) -> np.ndarray:
        """
        Pixelate a specific region
        
        Args:
            frame: Input frame
            x, y, w, h: Region coordinates
            pixel_size: Size of pixelation blocks
            
        Returns:
            Frame with pixelated region
        """
        output = frame.copy()
        
        # Extract region
        region = frame[y:y+h, x:x+w]
        
        # Resize down and up for pixelation effect
        height, width = region.shape[:2]
        small_w = max(1, width // pixel_size)
        small_h = max(1, height // pixel_size)
        
        small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Replace region
        output[y:y+h, x:x+w] = pixelated
        
        return output
    
    def black_box_region(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int
    ) -> np.ndarray:
        """
        Cover region with black box
        
        Args:
            frame: Input frame
            x, y, w, h: Region coordinates
            
        Returns:
            Frame with black box
        """
        output = frame.copy()
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 0), -1)
        return output
    
    def redact_faces(
        self,
        video_path: str,
        output_path: str,
        redaction_method: str = "blur",
        blur_intensity: int = 50,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Redact all faces in video
        
        Args:
            video_path: Input video path
            output_path: Output video path
            redaction_method: "blur", "pixelate", or "black"
            blur_intensity: Intensity for blur/pixelate
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with redaction statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_faces_redacted = 0
        frames_with_faces = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            if faces:
                frames_with_faces += 1
                total_faces_redacted += len(faces)
                
                # Redact each face
                for (x, y, w, h) in faces:
                    if redaction_method == "blur":
                        frame = self.blur_region(frame, x, y, w, h, blur_intensity)
                    elif redaction_method == "pixelate":
                        frame = self.pixelate_region(frame, x, y, w, h, blur_intensity)
                    elif redaction_method == "black":
                        frame = self.black_box_region(frame, x, y, w, h)
            
            # Write frame
            out.write(frame)
            
            # Progress callback
            if progress_callback and frame_count % 30 == 0:
                progress = frame_count / total_frames
                progress_callback(progress)
        
        cap.release()
        out.release()
        
        logger.info(f"Redacted {total_faces_redacted} faces in {frames_with_faces} frames")
        
        return {
            "total_frames": frame_count,
            "frames_with_faces": frames_with_faces,
            "total_faces_redacted": total_faces_redacted,
            "output_path": output_path
        }
    
    def redact_objects(
        self,
        video_path: str,
        output_path: str,
        detections: List[Dict],
        object_classes: List[str],
        redaction_method: str = "blur",
        blur_intensity: int = 50
    ) -> Dict:
        """
        Redact specific object classes in video
        
        Args:
            video_path: Input video path
            output_path: Output video path
            detections: Pre-computed object detections
            object_classes: Classes to redact
            redaction_method: "blur", "pixelate", or "black"
            blur_intensity: Intensity for blur/pixelate
            
        Returns:
            Dictionary with redaction statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Group detections by frame
        frame_detections = {}
        for det in detections:
            if det["class_name"] in object_classes:
                frame_num = det.get("frame_number", 0)
                if frame_num not in frame_detections:
                    frame_detections[frame_num] = []
                frame_detections[frame_num].append(det)
        
        frame_count = 0
        objects_redacted = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if frame has objects to redact
            if frame_count in frame_detections:
                for det in frame_detections[frame_count]:
                    bbox = det["bbox"]
                    x1, y1, x2, y2 = map(int, bbox)
                    w = x2 - x1
                    h = y2 - y1
                    
                    if redaction_method == "blur":
                        frame = self.blur_region(frame, x1, y1, w, h, blur_intensity)
                    elif redaction_method == "pixelate":
                        frame = self.pixelate_region(frame, x1, y1, w, h, blur_intensity)
                    elif redaction_method == "black":
                        frame = self.black_box_region(frame, x1, y1, w, h)
                    
                    objects_redacted += 1
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        logger.info(f"Redacted {objects_redacted} objects across {len(frame_detections)} frames")
        
        return {
            "total_frames": frame_count,
            "frames_with_redactions": len(frame_detections),
            "objects_redacted": objects_redacted,
            "output_path": output_path
        }
    
    def redact_custom_regions(
        self,
        video_path: str,
        output_path: str,
        regions: List[Dict],
        redaction_method: str = "blur",
        blur_intensity: int = 50
    ) -> Dict:
        """
        Redact custom regions specified by user
        
        Args:
            video_path: Input video path
            output_path: Output video path
            regions: List of region dictionaries with frame_number, x, y, w, h
            redaction_method: "blur", "pixelate", or "black"
            blur_intensity: Intensity for blur/pixelate
            
        Returns:
            Dictionary with redaction statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Group regions by frame
        frame_regions = {}
        for region in regions:
            frame_num = region["frame_number"]
            if frame_num not in frame_regions:
                frame_regions[frame_num] = []
            frame_regions[frame_num].append(region)
        
        frame_count = 0
        regions_redacted = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Redact regions for this frame
            if frame_count in frame_regions:
                for region in frame_regions[frame_count]:
                    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                    
                    if redaction_method == "blur":
                        frame = self.blur_region(frame, x, y, w, h, blur_intensity)
                    elif redaction_method == "pixelate":
                        frame = self.pixelate_region(frame, x, y, w, h, blur_intensity)
                    elif redaction_method == "black":
                        frame = self.black_box_region(frame, x, y, w, h)
                    
                    regions_redacted += 1
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        logger.info(f"Redacted {regions_redacted} custom regions")
        
        return {
            "total_frames": frame_count,
            "regions_redacted": regions_redacted,
            "output_path": output_path
        }
    
    def preview_redaction(
        self,
        frame: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
        redaction_method: str = "blur",
        blur_intensity: int = 50
    ) -> np.ndarray:
        """
        Preview redaction on a single frame
        
        Args:
            frame: Input frame
            regions: List of (x, y, w, h) tuples
            redaction_method: "blur", "pixelate", or "black"
            blur_intensity: Intensity for blur/pixelate
            
        Returns:
            Frame with redactions applied
        """
        output = frame.copy()
        
        for (x, y, w, h) in regions:
            if redaction_method == "blur":
                output = self.blur_region(output, x, y, w, h, blur_intensity)
            elif redaction_method == "pixelate":
                output = self.pixelate_region(output, x, y, w, h, blur_intensity)
            elif redaction_method == "black":
                output = self.black_box_region(output, x, y, w, h)
        
        return output
