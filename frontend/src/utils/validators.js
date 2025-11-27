export const validateVideoFile = (file) => {
  const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
  const maxSize = 500 * 1024 * 1024; // 500MB

  if (!file) {
    return { valid: false, error: 'No file provided' };
  }

  if (!validTypes.includes(file.type)) {
    return { valid: false, error: 'Invalid file type. Please use MP4, AVI, MOV, or MKV.' };
  }

  if (file.size > maxSize) {
    return { valid: false, error: 'File size exceeds 500MB limit.' };
  }

  return { valid: true };
};

export const validateEmail = (email) => {
  const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return regex.test(email);
};

export const validateURL = (url) => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};
