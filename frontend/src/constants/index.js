// Constants for frontend configuration

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
export const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000';

export const VIDEO_FORMATS = {
  ALLOWED_TYPES: ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/webm'],
  ALLOWED_EXTENSIONS: ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
  MAX_SIZE: 500 * 1024 * 1024, // 500MB
  MAX_DURATION: 3600, // 1 hour
};

export const DETECTION_CLASSES = [
  'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
  'dog', 'cat', 'bird', 'horse', 'sheep', 'cow',
  'chair', 'table', 'laptop', 'phone', 'book', 'bottle',
];

export const REDACTION_METHODS = {
  BLUR: 'blur',
  PIXELATE: 'pixelate',
  BLACK_BOX: 'black',
};

export const TIMELINE_EVENT_TYPES = {
  SCENE_CHANGE: 'scene_change',
  OBJECT_APPEARANCE: 'object_appearance',
  MOTION: 'motion',
  SPEECH: 'speech',
  SILENCE: 'silence',
};

export const PLAYBACK_SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];

export const THEME_OPTIONS = {
  LIGHT: 'light',
  DARK: 'dark',
  AUTO: 'auto',
};

export const ERROR_MESSAGES = {
  UPLOAD_FAILED: 'Failed to upload video. Please try again.',
  ANALYSIS_FAILED: 'Failed to analyze video. Please try again.',
  NETWORK_ERROR: 'Network error. Please check your connection.',
  FILE_TOO_LARGE: 'File size exceeds the maximum limit.',
  INVALID_FORMAT: 'Invalid video format. Please use a supported format.',
  VIDEO_NOT_FOUND: 'Video not found.',
};

export const SUCCESS_MESSAGES = {
  UPLOAD_SUCCESS: 'Video uploaded successfully!',
  ANALYSIS_COMPLETE: 'Analysis completed successfully!',
  REDACTION_COMPLETE: 'Video redaction completed!',
  DOWNLOAD_STARTED: 'Download started...',
};

export const POLLING_INTERVALS = {
  JOB_STATUS: 2000, // 2 seconds
  VIDEO_PROGRESS: 1000, // 1 second
};

export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 20,
  MAX_PAGE_SIZE: 100,
};
