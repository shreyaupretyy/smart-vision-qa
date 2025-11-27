import { useState } from 'react';
import { Clock, Loader2 } from 'lucide-react';
import { generateTimeline } from '../services/api';
import toast from 'react-hot-toast';

export default function TimelinePanel({ videoId }) {
  const [loading, setLoading] = useState(false);
  const [timeline, setTimeline] = useState(null);

  const handleGenerate = async () => {
    setLoading(true);
    setTimeline(null);

    try {
      const data = await generateTimeline(videoId);
      setTimeline(data);
      toast.success('Timeline generated');
    } catch (error) {
      console.error('Timeline error:', error);
      toast.error(error.response?.data?.detail || 'Failed to generate timeline');
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-white">Event Timeline</h3>
      
      <button
        onClick={handleGenerate}
        disabled={loading}
        className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Generating...
          </>
        ) : (
          <>
            <Clock className="w-4 h-4" />
            Generate Timeline
          </>
        )}
      </button>

      {timeline && (
        <div className="space-y-3">
          {timeline.summary && (
            <div className="bg-primary-600/10 border border-primary-600/20 rounded-lg p-3">
              <p className="text-sm text-gray-300">{timeline.summary}</p>
            </div>
          )}

          <div className="space-y-2 max-h-96 overflow-y-auto">
            {timeline.events.map((event, index) => (
              <div key={index} className="bg-gray-700/50 rounded-lg p-3 space-y-1">
                <div className="flex items-start justify-between">
                  <span className="text-primary-400 font-mono text-sm">
                    {formatTime(event.timestamp)}
                  </span>
                  <span className="text-xs text-gray-500 capitalize">
                    {event.event_type.replace('_', ' ')}
                  </span>
                </div>
                <p className="text-sm text-white">{event.description}</p>
                <div className="flex items-center gap-2 text-xs text-gray-400">
                  <span>Frame: {event.frame_number}</span>
                  <span>•</span>
                  <span>Confidence: {(event.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
