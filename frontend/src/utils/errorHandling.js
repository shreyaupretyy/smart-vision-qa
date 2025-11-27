// API error handling utilities

export class ApiError extends Error {
  constructor(message, status, data = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

/**
 * Handle API errors and return user-friendly messages
 */
export const handleApiError = (error) => {
  if (error.response) {
    // Server responded with error status
    const status = error.response.status;
    const message = error.response.data?.error || error.response.data?.message;
    
    switch (status) {
      case 400:
        return `Invalid request: ${message || 'Please check your input'}`;
      case 401:
        return 'Unauthorized: Please log in again';
      case 403:
        return 'Access denied: You do not have permission';
      case 404:
        return `Not found: ${message || 'The requested resource was not found'}`;
      case 413:
        return 'File too large: Please upload a smaller video';
      case 415:
        return 'Unsupported format: Please use a supported video format';
      case 422:
        return `Validation error: ${message || 'Invalid data provided'}`;
      case 429:
        return 'Too many requests: Please try again later';
      case 500:
        return 'Server error: Something went wrong on our end';
      case 503:
        return 'Service unavailable: Please try again later';
      default:
        return message || `Error: ${status}`;
    }
  } else if (error.request) {
    // Request made but no response
    return 'Network error: Unable to reach the server';
  } else {
    // Error setting up request
    return error.message || 'An unexpected error occurred';
  }
};

/**
 * Extract error message from various error formats
 */
export const getErrorMessage = (error) => {
  if (typeof error === 'string') {
    return error;
  }
  
  if (error instanceof ApiError) {
    return error.message;
  }
  
  if (error.response?.data?.error) {
    return error.response.data.error;
  }
  
  if (error.response?.data?.message) {
    return error.response.data.message;
  }
  
  if (error.message) {
    return error.message;
  }
  
  return 'An unknown error occurred';
};

/**
 * Check if error is a network error
 */
export const isNetworkError = (error) => {
  return !error.response && error.request;
};

/**
 * Check if error is a client error (4xx)
 */
export const isClientError = (error) => {
  return error.response && error.response.status >= 400 && error.response.status < 500;
};

/**
 * Check if error is a server error (5xx)
 */
export const isServerError = (error) => {
  return error.response && error.response.status >= 500;
};
