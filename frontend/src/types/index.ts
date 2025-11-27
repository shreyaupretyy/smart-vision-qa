// API response types
export interface ApiResponse<T = any> {
  data?: T;
  error?: string;
  message?: string;
  status: number;
}

export interface VideoMetadata {
  id: string;
  filename: string;
  duration: number;
  fps: number;
  width: number;
  height: number;
  frame_count: number;
  size: number;
  created_at: string;
}

export interface DetectionResult {
  frame_number: number;
  timestamp: number;
  detections: Detection[];
}

export interface Detection {
  class: string;
  confidence: number;
  bbox: [number, number, number, number];
  track_id?: number;
}

export interface TranscriptionResult {
  text: string;
  segments: TranscriptionSegment[];
  language: string;
}

export interface TranscriptionSegment {
  start: number;
  end: number;
  text: string;
}

export interface SearchResult {
  frame_number: number;
  timestamp: number;
  similarity: number;
  frame_path: string;
}

export interface TimelineEvent {
  start_time: number;
  end_time: number;
  event_type: string;
  description: string;
  confidence: number;
  thumbnail_path?: string;
}

export interface RedactionJob {
  job_id: string;
  video_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  output_path?: string;
  progress?: number;
}

export interface QueryResponse {
  answer: string;
  confidence: number;
  relevant_frames: number[];
}
