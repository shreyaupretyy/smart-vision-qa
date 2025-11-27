// Video-related utility functions

/**
 * Convert seconds to HH:MM:SS format
 */
export const formatTime = (seconds) => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  
  if (hours > 0) {
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

/**
 * Format video duration for display
 */
export const formatDuration = (seconds) => {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  } else if (seconds < 3600) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
  }
};

/**
 * Get frame number from timestamp
 */
export const getFrameNumber = (timestamp, fps) => {
  return Math.floor(timestamp * fps);
};

/**
 * Get timestamp from frame number
 */
export const getTimestamp = (frameNumber, fps) => {
  return frameNumber / fps;
};

/**
 * Format video resolution
 */
export const formatResolution = (width, height) => {
  return `${width}x${height}`;
};

/**
 * Get video quality label based on resolution
 */
export const getQualityLabel = (width, height) => {
  if (height >= 2160) return '4K';
  if (height >= 1440) return '2K';
  if (height >= 1080) return 'Full HD';
  if (height >= 720) return 'HD';
  return 'SD';
};

/**
 * Calculate video aspect ratio
 */
export const getAspectRatio = (width, height) => {
  const gcd = (a, b) => b === 0 ? a : gcd(b, a % b);
  const divisor = gcd(width, height);
  return `${width / divisor}:${height / divisor}`;
};
