// Context for managing application state

import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { getPreferences, savePreferences } from '../utils/storage';

const AppContext = createContext();

const initialState = {
  currentVideo: null,
  videos: [],
  preferences: getPreferences(),
  theme: 'light',
  loading: false,
  error: null,
};

const actionTypes = {
  SET_CURRENT_VIDEO: 'SET_CURRENT_VIDEO',
  ADD_VIDEO: 'ADD_VIDEO',
  REMOVE_VIDEO: 'REMOVE_VIDEO',
  SET_VIDEOS: 'SET_VIDEOS',
  UPDATE_PREFERENCES: 'UPDATE_PREFERENCES',
  SET_THEME: 'SET_THEME',
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  CLEAR_ERROR: 'CLEAR_ERROR',
};

const appReducer = (state, action) => {
  switch (action.type) {
    case actionTypes.SET_CURRENT_VIDEO:
      return { ...state, currentVideo: action.payload };
    
    case actionTypes.ADD_VIDEO:
      return { ...state, videos: [...state.videos, action.payload] };
    
    case actionTypes.REMOVE_VIDEO:
      return {
        ...state,
        videos: state.videos.filter(v => v.id !== action.payload),
        currentVideo: state.currentVideo?.id === action.payload ? null : state.currentVideo,
      };
    
    case actionTypes.SET_VIDEOS:
      return { ...state, videos: action.payload };
    
    case actionTypes.UPDATE_PREFERENCES:
      const newPreferences = { ...state.preferences, ...action.payload };
      savePreferences(newPreferences);
      return { ...state, preferences: newPreferences };
    
    case actionTypes.SET_THEME:
      return { ...state, theme: action.payload };
    
    case actionTypes.SET_LOADING:
      return { ...state, loading: action.payload };
    
    case actionTypes.SET_ERROR:
      return { ...state, error: action.payload };
    
    case actionTypes.CLEAR_ERROR:
      return { ...state, error: null };
    
    default:
      return state;
  }
};

export const AppProvider = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Apply theme on mount and when it changes
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', state.theme);
  }, [state.theme]);

  const value = {
    state,
    dispatch,
    actions: actionTypes,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within AppProvider');
  }
  return context;
};

export { actionTypes };
