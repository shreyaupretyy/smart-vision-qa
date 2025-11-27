// Local storage utilities

const STORAGE_KEYS = {
  RECENT_VIDEOS: 'smartvision_recent_videos',
  USER_PREFERENCES: 'smartvision_user_preferences',
  SEARCH_HISTORY: 'smartvision_search_history',
  THEME: 'smartvision_theme',
};

/**
 * Save data to local storage
 */
export const saveToStorage = (key, data) => {
  try {
    localStorage.setItem(key, JSON.stringify(data));
    return true;
  } catch (error) {
    console.error('Error saving to storage:', error);
    return false;
  }
};

/**
 * Load data from local storage
 */
export const loadFromStorage = (key, defaultValue = null) => {
  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : defaultValue;
  } catch (error) {
    console.error('Error loading from storage:', error);
    return defaultValue;
  }
};

/**
 * Remove item from local storage
 */
export const removeFromStorage = (key) => {
  try {
    localStorage.removeItem(key);
    return true;
  } catch (error) {
    console.error('Error removing from storage:', error);
    return false;
  }
};

/**
 * Clear all app data from local storage
 */
export const clearStorage = () => {
  try {
    Object.values(STORAGE_KEYS).forEach(key => {
      localStorage.removeItem(key);
    });
    return true;
  } catch (error) {
    console.error('Error clearing storage:', error);
    return false;
  }
};

/**
 * Save recent video
 */
export const saveRecentVideo = (videoData) => {
  const recent = loadFromStorage(STORAGE_KEYS.RECENT_VIDEOS, []);
  const updated = [videoData, ...recent.filter(v => v.id !== videoData.id)].slice(0, 10);
  return saveToStorage(STORAGE_KEYS.RECENT_VIDEOS, updated);
};

/**
 * Get recent videos
 */
export const getRecentVideos = () => {
  return loadFromStorage(STORAGE_KEYS.RECENT_VIDEOS, []);
};

/**
 * Save search query to history
 */
export const saveSearchQuery = (query) => {
  const history = loadFromStorage(STORAGE_KEYS.SEARCH_HISTORY, []);
  const updated = [query, ...history.filter(q => q !== query)].slice(0, 20);
  return saveToStorage(STORAGE_KEYS.SEARCH_HISTORY, updated);
};

/**
 * Get search history
 */
export const getSearchHistory = () => {
  return loadFromStorage(STORAGE_KEYS.SEARCH_HISTORY, []);
};

/**
 * Save user preferences
 */
export const savePreferences = (preferences) => {
  return saveToStorage(STORAGE_KEYS.USER_PREFERENCES, preferences);
};

/**
 * Get user preferences
 */
export const getPreferences = () => {
  return loadFromStorage(STORAGE_KEYS.USER_PREFERENCES, {
    autoplay: false,
    volume: 0.7,
    playbackSpeed: 1.0,
    theme: 'light',
  });
};

export { STORAGE_KEYS };
