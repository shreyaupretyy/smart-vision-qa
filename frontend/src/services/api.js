import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Video APIs
export const uploadVideo = async (file, onProgress) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/video/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      const percentCompleted = Math.round(
        (progressEvent.loaded * 100) / progressEvent.total
      );
      if (onProgress) onProgress(percentCompleted);
    },
  });

  return response.data;
};

export const getVideoMetadata = async (videoId) => {
  const response = await api.get(`/video/${videoId}`);
  return response.data;
};

export const deleteVideo = async (videoId) => {
  const response = await api.delete(`/video/${videoId}`);
  return response.data;
};

export const getFrame = async (videoId, frameNumber) => {
  const response = await api.get(`/video/${videoId}/frame/${frameNumber}`, {
    responseType: 'blob',
  });
  return URL.createObjectURL(response.data);
};

// Analysis APIs
export const queryVideo = async (videoId, question, startTime, endTime) => {
  const response = await api.post('/analyze/query', {
    video_id: videoId,
    question,
    start_time: startTime,
    end_time: endTime,
  });
  return response.data;
};

export const detectObjects = async (videoId, startTime, endTime, confidence) => {
  const response = await api.post('/analyze/detect', {
    video_id: videoId,
    start_time: startTime,
    end_time: endTime,
    confidence_threshold: confidence || 0.5,
  });
  return response.data;
};

export const transcribeVideo = async (videoId) => {
  const response = await api.post('/analyze/transcribe', {
    video_id: videoId,
  });
  return response.data;
};

export const generateTimeline = async (videoId) => {
  const response = await api.get(`/analyze/timeline/${videoId}`);
  return response.data;
};

export const searchVideo = async (videoId, query, topK) => {
  const response = await api.post('/analyze/search', {
    video_id: videoId,
    query,
    top_k: topK || 5,
  });
  return response.data;
};

// Redaction APIs
export const redactFaces = async (videoId, blurIntensity) => {
  const response = await api.post('/redact/faces', {
    video_id: videoId,
    redaction_type: 'faces',
    blur_intensity: blurIntensity || 50,
  });
  return response.data;
};

export const redactObjects = async (videoId, objectClasses, blurIntensity) => {
  const response = await api.post('/redact/objects', {
    video_id: videoId,
    redaction_type: 'objects',
    object_classes: objectClasses,
    blur_intensity: blurIntensity || 50,
  });
  return response.data;
};

export default api;
