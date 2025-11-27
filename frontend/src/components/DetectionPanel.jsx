import { useState } from 'react';
import { Eye, Loader2 } from 'lucide-react';
import { detectObjects } from '../services/api';
import toast from 'react-hot-toast';

export default function DetectionPanel({ videoId }) {
  const [loading, setLoading] = useState(false);
  const [detections, setDetections] = useState(null);
  const [confidence, setConfidence] = useState(0.5);

  const handleDetect = async () => {
    setLoading(true);
    setDetections(null);

    try {
      const data = await detectObjects(videoId, null, null, confidence);
      setDetections(data);
      toast.success(`Detected ${data.detections.length} objects`);
    } catch (error) {
      console.error('Detection error:', error);
      toast.error(error.response?.data?.detail || 'Failed to detect objects');
    } finally {
      setLoading(false);
    }
  };

  const objectCounts = detections?.detections.reduce((acc, det) => {
    acc[det.class_name] = (acc[det.class_name] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-white">Object Detection</h3>
      
      <div className="space-y-3">
        <div>
          <label className="block text-sm text-gray-400 mb-2">
            Confidence Threshold: {(confidence * 100).toFixed(0)}%
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
            className="w-full"
            disabled={loading}
          />
        </div>
        
        <button
          onClick={handleDetect}
          disabled={loading}
          className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Detecting...
            </>
          ) : (
            <>
              <Eye className="w-4 h-4" />
              Detect Objects
            </>
          )}
        </button>
      </div>

      {detections && (
        <div className="space-y-3">
          <div className="bg-gray-700/50 rounded-lg p-3">
            <p className="text-sm text-gray-400">
              Found {detections.detections.length} objects in {detections.total_frames} frames
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Processing time: {detections.processing_time.toFixed(2)}s
            </p>
          </div>

          {objectCounts && (
            <div>
              <p className="text-sm text-gray-400 mb-2">Object Summary:</p>
              <div className="space-y-2">
                {Object.entries(objectCounts)
                  .sort((a, b) => b[1] - a[1])
                  .map(([className, count]) => (
                    <div key={className} className="flex justify-between items-center bg-gray-700/50 rounded p-2">
                      <span className="text-white capitalize">{className}</span>
                      <span className="text-primary-400 font-medium">{count}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
