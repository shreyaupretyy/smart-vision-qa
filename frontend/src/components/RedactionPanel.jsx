import { useState } from 'react';
import { Shield, Loader2 } from 'lucide-react';
import { redactFaces, redactObjects } from '../services/api';
import toast from 'react-hot-toast';

export default function RedactionPanel({ videoId }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [redactionType, setRedactionType] = useState('faces');
  const [blurIntensity, setBlurIntensity] = useState(50);

  const handleRedact = async () => {
    setLoading(true);
    setResult(null);

    try {
      let data;
      if (redactionType === 'faces') {
        data = await redactFaces(videoId, blurIntensity);
      } else {
        // For objects, you'd need to specify which classes
        data = await redactObjects(videoId, ['person'], blurIntensity);
      }
      
      setResult(data);
      toast.success(`Redacted ${data.items_redacted} items`);
    } catch (error) {
      console.error('Redaction error:', error);
      toast.error(error.response?.data?.detail || 'Failed to redact video');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-white">Video Redaction</h3>
      
      <div className="space-y-3">
        <div>
          <label className="block text-sm text-gray-400 mb-2">Redaction Type</label>
          <select
            value={redactionType}
            onChange={(e) => setRedactionType(e.target.value)}
            className="input-field"
            disabled={loading}
          >
            <option value="faces">Faces</option>
            <option value="objects">Objects (People)</option>
          </select>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-2">
            Blur Intensity: {blurIntensity}
          </label>
          <input
            type="range"
            min="10"
            max="100"
            step="10"
            value={blurIntensity}
            onChange={(e) => setBlurIntensity(parseInt(e.target.value))}
            className="w-full"
            disabled={loading}
          />
        </div>
        
        <button
          onClick={handleRedact}
          disabled={loading}
          className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Redacting...
            </>
          ) : (
            <>
              <Shield className="w-4 h-4" />
              Redact Video
            </>
          )}
        </button>
      </div>

      {result && (
        <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-4 space-y-2">
          <p className="text-green-400 font-medium">✓ Redaction Complete</p>
          <p className="text-sm text-gray-300">
            Redacted {result.items_redacted} {result.redaction_type}
          </p>
          <p className="text-xs text-gray-400 break-all">
            Output: {result.redacted_video_id}
          </p>
        </div>
      )}
    </div>
  );
}
