"""
Database configuration and session management with comprehensive models.
"""
from typing import Generator
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime
import os

# Create Base class for models
Base = declarative_base()


class Video(Base):
    """Video metadata and processing status"""
    __tablename__ = "videos"
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    status = Column(String, nullable=False, default="uploading")
    fps = Column(Float)
    frame_count = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    duration = Column(Float)
    file_size = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    frames = relationship("Frame", back_populates="video", cascade="all, delete-orphan")
    queries = relationship("Query", back_populates="video", cascade="all, delete-orphan")
    detections = relationship("Detection", back_populates="video", cascade="all, delete-orphan")
    metrics = relationship("VideoMetrics", back_populates="video", uselist=False, cascade="all, delete-orphan")


class Frame(Base):
    """Individual video frames with embeddings and captions"""
    __tablename__ = "frames"
    
    id = Column(Integer, primary_key=True)
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    caption = Column(Text, nullable=True)
    embedding_vector = Column(Text, nullable=True)  # Store as comma-separated string for SQLite
    extracted_at = Column(DateTime, default=datetime.utcnow)
    
    video = relationship("Video", back_populates="frames")


class Query(Base):
    """Query history with answers and performance metrics"""
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True)
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    confidence = Column(Float)
    relevant_frames = Column(Text)  # Comma-separated frame numbers
    processing_time = Column(Float)  # In seconds
    model_used = Column(String, default="blip-vqa-base")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    video = relationship("Video", back_populates="queries")


class Detection(Base):
    """Object detection results"""
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True)
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    object_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    detected_at = Column(DateTime, default=datetime.utcnow)
    
    video = relationship("Video", back_populates="detections")


class VideoMetrics(Base):
    """Performance metrics for video processing"""
    __tablename__ = "video_metrics"
    
    id = Column(Integer, primary_key=True)
    video_id = Column(String, ForeignKey("videos.id"), nullable=False, unique=True)
    
    # Processing metrics
    frame_extraction_time = Column(Float)
    caption_generation_time = Column(Float)
    embedding_generation_time = Column(Float)
    total_processing_time = Column(Float)
    
    # Quality metrics
    average_caption_length = Column(Float)
    unique_objects_detected = Column(Integer, default=0)
    total_queries = Column(Integer, default=0)
    average_query_time = Column(Float)
    average_confidence = Column(Float)
    
    # Resource metrics
    peak_memory_usage = Column(Float)  # In MB
    gpu_used = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    video = relationship("Video", back_populates="metrics")


class ModelComparison(Base):
    """Track different model performance for A/B testing"""
    __tablename__ = "model_comparisons"
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # 'vqa', 'caption', 'detection'
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    confidence = Column(Float)
    processing_time = Column(Float)
    accuracy_score = Column(Float, nullable=True)  # If ground truth available
    created_at = Column(DateTime, default=datetime.utcnow)


# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./smartvisionqa.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,
    echo=os.getenv("DEBUG", "True") == "True"
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for getting database session.
    
    Yields:
        Session: Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database tables.
    Creates all tables defined in models.
    """
    Base.metadata.create_all(bind=engine)
    print("✓ Database initialized successfully")


def drop_db() -> None:
    """Drop all database tables (for testing)"""
    Base.metadata.drop_all(bind=engine)
    print("✓ Database dropped successfully")
