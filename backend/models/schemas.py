from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class VideoStatus(str, Enum):
    """Video processing status"""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class VideoUploadResponse(BaseModel):
    """Response for video upload"""
    video_id: str
    filename: str
    size: int
    status: VideoStatus
    message: str


class VideoMetadata(BaseModel):
    """Video metadata"""
    video_id: str
    filename: str
    size: int
    duration: float
    fps: float
    width: int
    height: int
    frame_count: int
    status: VideoStatus
    upload_time: datetime
    processed_time: Optional[datetime] = None


class QueryRequest(BaseModel):
    """Request for video query"""
    video_id: str
    question: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class QueryResponse(BaseModel):
    """Response for video query"""
    answer: str
    confidence: float
    relevant_frames: List[int]
    timestamp: Optional[float] = None


class DetectionRequest(BaseModel):
    """Request for object detection"""
    video_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence_threshold: float = 0.5


class Detection(BaseModel):
    """Single detection result"""
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    frame_number: int
    timestamp: float


class DetectionResponse(BaseModel):
    """Response for object detection"""
    video_id: str
    detections: List[Detection]
    total_frames: int
    processing_time: float


class TranscriptionRequest(BaseModel):
    """Request for audio transcription"""
    video_id: str


class TranscriptionSegment(BaseModel):
    """Transcription segment"""
    text: str
    start: float
    end: float
    confidence: float


class TranscriptionResponse(BaseModel):
    """Response for transcription"""
    video_id: str
    language: str
    segments: List[TranscriptionSegment]
    full_text: str


class TimelineEvent(BaseModel):
    """Timeline event"""
    timestamp: float
    event_type: str
    description: str
    confidence: float
    frame_number: int


class TimelineResponse(BaseModel):
    """Response for timeline generation"""
    video_id: str
    events: List[TimelineEvent]
    summary: str


class SearchRequest(BaseModel):
    """Request for semantic search"""
    video_id: str
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    """Search result"""
    frame_number: int
    timestamp: float
    similarity: float
    description: str


class SearchResponse(BaseModel):
    """Response for semantic search"""
    video_id: str
    query: str
    results: List[SearchResult]


class RedactionRequest(BaseModel):
    """Request for video redaction"""
    video_id: str
    redaction_type: str  # "faces", "objects", "custom"
    object_classes: Optional[List[str]] = None
    blur_intensity: int = 50


class RedactionResponse(BaseModel):
    """Response for redaction"""
    video_id: str
    redacted_video_id: str
    redaction_type: str
    items_redacted: int
    output_path: str


class AnnotationCreate(BaseModel):
    """Create annotation"""
    video_id: str
    frame_number: int
    timestamp: float
    annotation_type: str
    data: Dict[str, Any]
    user_id: str


class Annotation(BaseModel):
    """Annotation model"""
    id: str
    video_id: str
    frame_number: int
    timestamp: float
    annotation_type: str
    data: Dict[str, Any]
    user_id: str
    created_at: datetime


class WebSocketMessage(BaseModel):
    """WebSocket message"""
    type: str
    data: Dict[str, Any]
    user_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
