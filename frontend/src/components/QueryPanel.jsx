import { useState } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { queryVideo } from '../services/api';
import toast from 'react-hot-toast';

export default function QueryPanel({ videoId }) {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const data = await queryVideo(videoId, question);
      setResult(data);
    } catch (error) {
      console.error('Query error:', error);
      toast.error(error.response?.data?.detail || 'Failed to process question');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-white">Ask Questions</h3>
      
      <form onSubmit={handleSubmit} className="space-y-3">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g., What objects are visible in the video?"
          className="input-field min-h-[100px] resize-none"
          disabled={loading}
        />
        
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Send className="w-4 h-4" />
              Ask Question
            </>
          )}
        </button>
      </form>

      {result && (
        <div className="bg-gray-700/50 rounded-lg p-4 space-y-3">
          <div>
            <p className="text-sm text-gray-400 mb-1">Answer:</p>
            <p className="text-white">{result.answer}</p>
          </div>
          
          <div className="flex items-center gap-4 text-sm">
            <span className="text-gray-400">
              Confidence: <span className="text-white">{(result.confidence * 100).toFixed(1)}%</span>
            </span>
            {result.timestamp && (
              <span className="text-gray-400">
                At: <span className="text-white">{result.timestamp.toFixed(1)}s</span>
              </span>
            )}
          </div>

          {result.relevant_frames && result.relevant_frames.length > 0 && (
            <div>
              <p className="text-sm text-gray-400 mb-2">Relevant Frames:</p>
              <div className="flex flex-wrap gap-2">
                {result.relevant_frames.map((frame) => (
                  <span key={frame} className="px-2 py-1 bg-primary-600/20 text-primary-400 rounded text-sm">
                    #{frame}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
