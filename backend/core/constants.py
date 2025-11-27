"""
Constants used across the application
"""

# Supported video formats
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

# Model names
YOLO_MODELS = {
    'nano': 'yolov8n.pt',
    'small': 'yolov8s.pt',
    'medium': 'yolov8m.pt',
    'large': 'yolov8l.pt',
    'xlarge': 'yolov8x.pt',
}

WHISPER_MODELS = ['tiny', 'base', 'small', 'medium', 'large']

# Processing limits
MAX_VIDEO_DURATION = 3600  # 1 hour
MAX_FRAME_RATE = 60  # FPS
MIN_FRAME_RATE = 1  # FPS

# Detection settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.45

# Redaction settings
BLUR_METHODS = ['blur', 'pixelate', 'black']
DEFAULT_BLUR_INTENSITY = 50

# WebSocket event types
WS_EVENT_USER_JOINED = 'user_joined'
WS_EVENT_USER_LEFT = 'user_left'
WS_EVENT_ANNOTATION = 'annotation'
WS_EVENT_CURSOR = 'cursor'
WS_EVENT_CHAT = 'chat'
WS_EVENT_PLAYBACK = 'playback'
