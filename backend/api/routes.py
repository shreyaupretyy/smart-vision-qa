from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from typing import List, Optional
import logging
from pathlib import Path
import uuid
import asyncio

from backend.models.schemas import *
from backend.core.config import get_settings
from backend.services.video_processor import VideoProcessor
from backend.services.object_detection import ObjectDetector
from backend.services.vision_qa import VisionQA
from backend.services.transcription import AudioTranscriber
from backend.services.embeddings import EmbeddingsService
from backend.services.timeline import TimelineGenerator
from backend.services.redaction import VideoRedactor

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services (singleton pattern)
settings = get_settings()
video_processor = VideoProcessor(settings.upload_dir, settings.temp_dir)
object_detector = None
vision_qa = None
transcriber = None
embeddings_service = None
timeline_generator = TimelineGenerator()
video_redactor = VideoRedactor()

# In-memory video metadata storage (use database in production)
video_metadata_db = {}


def get_object_detector():
    """Lazy load object detector"""
    global object_detector
    if object_detector is None:
        object_detector = ObjectDetector(settings.yolo_model)
    return object_detector


def get_vision_qa():
    """Lazy load vision QA"""
    global vision_qa
    if vision_qa is None:
        vision_qa = VisionQA(settings.blip_model)
    return vision_qa


def get_transcriber():
    """Lazy load transcriber"""
    global transcriber
    if transcriber is None:
        transcriber = AudioTranscriber(settings.whisper_model)
    return transcriber


def get_embeddings_service():
    """Lazy load embeddings service"""
    global embeddings_service
    if embeddings_service is None:
        embeddings_service = EmbeddingsService(settings.chroma_persist_dir)
    return embeddings_service


async def process_video_background(video_id: str, video_path: str):
    """Background task to process uploaded video"""
    try:
        logger.info(f"Processing video {video_id}")
        
        # Update status
        video_metadata_db[video_id]["status"] = VideoStatus.PROCESSING
        
        # Extract frames
        frames = video_processor.extract_frames(
            video_path,
            sample_rate=settings.frame_sample_rate
        )
        
        logger.info(f"Extracted {len(frames)} frames")
        
        # Generate captions for frames
        qa = get_vision_qa()
        frames_data = []
        
        for frame_num, timestamp, frame in frames:
            caption = qa.generate_caption(frame)
            frames_data.append({
                "frame_number": frame_num,
                "timestamp": timestamp,
                "caption": caption
            })
        
        # Store embeddings
        embeddings = get_embeddings_service()
        embeddings.batch_add_frames(video_id, frames_data)
        
        # Update status
        video_metadata_db[video_id]["status"] = VideoStatus.READY
        video_metadata_db[video_id]["processed_time"] = datetime.utcnow()
        
        logger.info(f"Video {video_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
        video_metadata_db[video_id]["status"] = VideoStatus.FAILED


# ==================== VIDEO ENDPOINTS ====================

@router.post("/video/upload", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload a video file"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.mp4', '.avi', '.mov', '.mkv']:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Use MP4, AVI, MOV, or MKV"
            )
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.max_upload_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.max_upload_size} bytes"
            )
        
        # Save video
        video_id, file_path = video_processor.save_uploaded_video(content, file.filename)
        
        # Get metadata
        metadata = video_processor.get_video_metadata(file_path)
        
        # Store metadata
        video_metadata_db[video_id] = {
            "video_id": video_id,
            "filename": file.filename,
            "size": file_size,
            "duration": metadata["duration"],
            "fps": metadata["fps"],
            "width": metadata["width"],
            "height": metadata["height"],
            "frame_count": metadata["frame_count"],
            "status": VideoStatus.UPLOADING,
            "upload_time": datetime.utcnow(),
            "processed_time": None
        }
        
        # Process video in background
        background_tasks.add_task(process_video_background, video_id, file_path)
        
        return VideoUploadResponse(
            video_id=video_id,
            filename=file.filename,
            size=file_size,
            status=VideoStatus.PROCESSING,
            message="Video uploaded successfully. Processing in background."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/video/{video_id}", response_model=VideoMetadata)
async def get_video_metadata(video_id: str):
    """Get video metadata"""
    if video_id not in video_metadata_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return video_metadata_db[video_id]


@router.delete("/video/{video_id}")
async def delete_video(video_id: str):
    """Delete a video"""
    if video_id not in video_metadata_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Delete video file
    video_processor.delete_video(video_id)
    
    # Delete embeddings
    embeddings = get_embeddings_service()
    embeddings.delete_video_frames(video_id)
    
    # Remove from metadata
    del video_metadata_db[video_id]
    
    return {"message": "Video deleted successfully"}


@router.get("/video/{video_id}/frame/{frame_number}")
async def get_frame(video_id: str, frame_number: int):
    """Get a specific frame from video"""
    if video_id not in video_metadata_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = video_processor.get_video_path(video_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video file not found")
    
    metadata = video_metadata_db[video_id]
    fps = metadata["fps"]
    timestamp = frame_number / fps
    
    frame = video_processor.get_frame_at_time(video_path, timestamp)
    
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame not found")
    
    # Save frame temporarily
    temp_path = Path(settings.temp_dir) / f"{video_id}_frame_{frame_number}.jpg"
    video_processor.save_frame(frame, str(temp_path))
    
    return FileResponse(temp_path, media_type="image/jpeg")


# ==================== ANALYSIS ENDPOINTS ====================

@router.post("/analyze/query", response_model=QueryResponse)
async def query_video(request: QueryRequest):
    """Ask a question about the video"""
    if request.video_id not in video_metadata_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video_metadata_db[request.video_id]["status"] != VideoStatus.READY:
        raise HTTPException(
            status_code=400,
            detail="Video not ready. Please wait for processing to complete."
        )
    
    try:
        video_path = video_processor.get_video_path(request.video_id)
        if not video_path:
            raise HTTPException(status_code=404, detail="Video file not found")
        
        qa = get_vision_qa()
        
        # Query video
        result = qa.query_video(
            video_path,
            request.question,
            sample_rate=2,
            top_k=3
        )
        
        return QueryResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            relevant_frames=result["relevant_frames"],
            timestamp=result["detailed_results"][0]["timestamp"] if result["detailed_results"] else None
        )
        
    except Exception as e:
        logger.error(f"Error querying video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/detect", response_model=DetectionResponse)
async def detect_objects(request: DetectionRequest):
    """Detect objects in video"""
    if request.video_id not in video_metadata_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        video_path = video_processor.get_video_path(request.video_id)
        if not video_path:
            raise HTTPException(status_code=404, detail="Video file not found")
        
        detector = get_object_detector()
        
        # Get video metadata
        metadata = video_metadata_db[request.video_id]
        fps = metadata["fps"]
        
        # Calculate frame range
        start_frame = int(request.start_time * fps) if request.start_time else 0
        end_frame = int(request.end_time * fps) if request.end_time else None
        
        # Detect objects
        import time
        start_time = time.time()
        
        detections = detector.detect_in_video(
            video_path,
            confidence_threshold=request.confidence_threshold,
            start_frame=start_frame,
            end_frame=end_frame,
            sample_rate=int(fps)  # 1 frame per second
        )
        
        processing_time = time.time() - start_time
        
        # Convert to response format
        detection_list = [
            Detection(
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=d["bbox"],
                frame_number=d["frame_number"],
                timestamp=d["timestamp"]
            )
            for d in detections
        ]
        
        return DetectionResponse(
            video_id=request.video_id,
            detections=detection_list,
            total_frames=metadata["frame_count"],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error detecting objects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/transcribe", response_model=TranscriptionResponse)
async def transcribe_video(request: TranscriptionRequest):
    """Transcribe audio from video"""
    if request.video_id not in video_metadata_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        video_path = video_processor.get_video_path(request.video_id)
        if not video_path:
            raise HTTPException(status_code=404, detail="Video file not found")
        
        transcriber = get_transcriber()
        
        # Transcribe
        result = transcriber.transcribe_video(video_path)
        
        # Convert segments
        segments = [
            TranscriptionSegment(
                text=seg["text"],
                start=seg["start"],
                end=seg["end"],
                confidence=seg["confidence"]
            )
            for seg in result["segments"]
        ]
        
        return TranscriptionResponse(
            video_id=request.video_id,
            language=result["language"],
            segments=segments,
            full_text=result["text"]
        )
        
    except Exception as e:
        logger.error(f"Error transcribing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/timeline/{video_id}", response_model=TimelineResponse)
async def generate_timeline(video_id: str):
    """Generate event timeline for video"""
    if video_id not in video_metadata_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video_metadata_db[video_id]["status"] != VideoStatus.READY:
        raise HTTPException(
            status_code=400,
            detail="Video not ready. Please wait for processing to complete."
        )
    
    try:
        # Get stored data
        embeddings = get_embeddings_service()
        
        # For now, create simplified timeline from embeddings
        # In production, you'd retrieve actual detections and captions
        
        timeline_data = timeline_generator.generate_timeline(
            video_id,
            detections=[],  # Would be retrieved from storage
            captions=[],  # Would be retrieved from embeddings
            transcripts=None
        )
        
        events = [
            TimelineEvent(
                timestamp=e["timestamp"],
                event_type=e["event_type"],
                description=e["description"],
                confidence=e["confidence"],
                frame_number=e["frame_number"]
            )
            for e in timeline_data["events"]
        ]
        
        return TimelineResponse(
            video_id=video_id,
            events=events,
            summary=timeline_data["summary"]
        )
        
    except Exception as e:
        logger.error(f"Error generating timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/search", response_model=SearchResponse)
async def search_video(request: SearchRequest):
    """Semantic search in video"""
    if request.video_id not in video_metadata_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video_metadata_db[request.video_id]["status"] != VideoStatus.READY:
        raise HTTPException(
            status_code=400,
            detail="Video not ready. Please wait for processing to complete."
        )
    
    try:
        embeddings = get_embeddings_service()
        
        # Search frames
        matches = embeddings.search_frames(
            query=request.query,
            video_id=request.video_id,
            top_k=request.top_k,
            min_similarity=0.3
        )
        
        # Convert to response format
        results = [
            SearchResult(
                frame_number=match["metadata"]["frame_number"],
                timestamp=match["metadata"]["timestamp"],
                similarity=match["similarity"],
                description=match["metadata"].get("caption", "")
            )
            for match in matches
        ]
        
        return SearchResponse(
            video_id=request.video_id,
            query=request.query,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Error searching video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== REDACTION ENDPOINTS ====================

@router.post("/redact/faces", response_model=RedactionResponse)
async def redact_faces(request: RedactionRequest):
    """Redact faces in video"""
    if request.video_id not in video_metadata_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        video_path = video_processor.get_video_path(request.video_id)
        if not video_path:
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Generate output path
        redacted_video_id = str(uuid.uuid4())
        output_path = str(Path(settings.upload_dir) / f"{redacted_video_id}.mp4")
        
        # Redact faces
        result = video_redactor.redact_faces(
            video_path,
            output_path,
            redaction_method="blur",
            blur_intensity=request.blur_intensity
        )
        
        return RedactionResponse(
            video_id=request.video_id,
            redacted_video_id=redacted_video_id,
            redaction_type="faces",
            items_redacted=result["total_faces_redacted"],
            output_path=output_path
        )
        
    except Exception as e:
        logger.error(f"Error redacting faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/redact/objects", response_model=RedactionResponse)
async def redact_objects(request: RedactionRequest):
    """Redact specific objects in video"""
    if request.video_id not in video_metadata_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not request.object_classes:
        raise HTTPException(status_code=400, detail="No object classes specified")
    
    try:
        video_path = video_processor.get_video_path(request.video_id)
        if not video_path:
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # First detect objects
        detector = get_object_detector()
        detections = detector.detect_in_video(video_path, confidence_threshold=0.5)
        
        # Generate output path
        redacted_video_id = str(uuid.uuid4())
        output_path = str(Path(settings.upload_dir) / f"{redacted_video_id}.mp4")
        
        # Redact objects
        result = video_redactor.redact_objects(
            video_path,
            output_path,
            detections,
            request.object_classes,
            redaction_method="blur",
            blur_intensity=request.blur_intensity
        )
        
        return RedactionResponse(
            video_id=request.video_id,
            redacted_video_id=redacted_video_id,
            redaction_type="objects",
            items_redacted=result["objects_redacted"],
            output_path=output_path
        )
        
    except Exception as e:
        logger.error(f"Error redacting objects: {e}")
        raise HTTPException(status_code=500, detail=str(e))
