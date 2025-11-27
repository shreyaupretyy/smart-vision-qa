import { Upload, X } from 'lucide-react';
import { useState, useCallback } from 'react';
import { uploadVideo } from '../services/api';
import toast from 'react-hot-toast';

export default function VideoUpload({ onUploadComplete }) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, []);

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleFileSelect = (file) => {
    // Validate file type
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
    if (!validTypes.includes(file.type)) {
      toast.error('Please select a valid video file (MP4, AVI, MOV, MKV)');
      return;
    }

    // Validate file size (500MB max)
    if (file.size > 500 * 1024 * 1024) {
      toast.error('File size must be less than 500MB');
      return;
    }

    setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadProgress(0);

    try {
      const result = await uploadVideo(selectedFile, (progress) => {
        setUploadProgress(progress);
      });

      toast.success('Video uploaded successfully!');
      setSelectedFile(null);
      setUploadProgress(0);

      if (onUploadComplete) {
        onUploadComplete(result);
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast.error(error.response?.data?.detail || 'Failed to upload video');
    } finally {
      setUploading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setUploadProgress(0);
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-bold text-white mb-6">Upload Video</h2>

      {!selectedFile ? (
        <div
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${
            isDragging
              ? 'border-primary-500 bg-primary-500/10'
              : 'border-gray-600 hover:border-gray-500'
          }`}
        >
          <input
            type="file"
            id="video-upload"
            accept="video/mp4,video/avi,video/quicktime,video/x-matroska"
            onChange={handleFileInput}
            className="hidden"
          />
          <label htmlFor="video-upload" className="cursor-pointer">
            <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
            <p className="text-lg text-gray-300 mb-2">
              {isDragging ? 'Drop video here' : 'Click to upload or drag and drop'}
            </p>
            <p className="text-sm text-gray-500">
              Supports MP4, AVI, MOV, MKV (max 500MB)
            </p>
          </label>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
            <div className="flex-1">
              <p className="text-white font-medium truncate">{selectedFile.name}</p>
              <p className="text-sm text-gray-400">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
            {!uploading && (
              <button
                onClick={handleClear}
                className="ml-4 p-2 hover:bg-gray-600 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-400" />
              </button>
            )}
          </div>

          {uploading && (
            <div>
              <div className="flex justify-between text-sm text-gray-400 mb-2">
                <span>Uploading...</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={uploading}
            className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {uploading ? 'Uploading...' : 'Upload Video'}
          </button>
        </div>
      )}
    </div>
  );
}
