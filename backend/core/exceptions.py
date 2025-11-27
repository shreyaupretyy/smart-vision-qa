"""
Custom exceptions for SmartVisionQA
"""


class SmartVisionQAException(Exception):
    """Base exception for SmartVisionQA"""
    pass


class VideoProcessingError(SmartVisionQAException):
    """Error during video processing"""
    pass


class ModelLoadError(SmartVisionQAException):
    """Error loading AI model"""
    pass


class VideoNotFoundError(SmartVisionQAException):
    """Video not found"""
    pass


class InvalidVideoFormat(SmartVisionQAException):
    """Invalid video format"""
    pass


class FileSizeLimitExceeded(SmartVisionQAException):
    """File size exceeds limit"""
    pass


class EmbeddingError(SmartVisionQAException):
    """Error creating or querying embeddings"""
    pass


class TranscriptionError(SmartVisionQAException):
    """Error during audio transcription"""
    pass


class DetectionError(SmartVisionQAException):
    """Error during object detection"""
    pass


class RedactionError(SmartVisionQAException):
    """Error during video redaction"""
    pass
