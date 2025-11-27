import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  MessageSquare,
  Search,
  Eye,
  Clock,
  Shield,
  Loader2,
} from 'lucide-react';
import toast from 'react-hot-toast';
import {
  getVideoMetadata,
  queryVideo,
  searchVideo,
  detectObjects,
  generateTimeline,
  redactFaces,
} from '../services/api';
import VideoPlayer from '../components/VideoPlayer';
import QueryPanel from '../components/QueryPanel';
import SearchPanel from '../components/SearchPanel';
import DetectionPanel from '../components/DetectionPanel';
import TimelinePanel from '../components/TimelinePanel';
import RedactionPanel from '../components/RedactionPanel';

export default function AnalyzePage() {
  const { videoId } = useParams();
  const [metadata, setMetadata] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('query');

  useEffect(() => {
    loadMetadata();
    const interval = setInterval(loadMetadata, 3000); // Poll for updates
    return () => clearInterval(interval);
  }, [videoId]);

  const loadMetadata = async () => {
    try {
      const data = await getVideoMetadata(videoId);
      setMetadata(data);
      setLoading(false);
    } catch (error) {
      console.error('Error loading metadata:', error);
      toast.error('Failed to load video metadata');
      setLoading(false);
    }
  };

  const tabs = [
    { id: 'query', label: 'Ask Questions', icon: MessageSquare },
    { id: 'search', label: 'Search', icon: Search },
    { id: 'detect', label: 'Detect Objects', icon: Eye },
    { id: 'timeline', label: 'Timeline', icon: Clock },
    { id: 'redact', label: 'Redact', icon: Shield },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-primary-400 animate-spin mx-auto mb-4" />
          <p className="text-lg text-gray-300">Loading video...</p>
        </div>
      </div>
    );
  }

  if (!metadata) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
        <div className="text-center">
          <p className="text-xl text-gray-300 mb-2">Video not found</p>
          <p className="text-sm text-gray-500">The video may have been deleted or the ID is invalid</p>
        </div>
      </div>
    );
  }

  const isProcessing = metadata.status === 'processing' || metadata.status === 'uploading';

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Video Info Header */}
        <div className="bg-gradient-to-r from-gray-900/80 to-gray-800/80 backdrop-blur-sm rounded-2xl shadow-2xl border border-gray-700/50 mb-8 p-6">
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
              <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-purple-400 mb-3">
                {metadata.filename}
              </h1>
              <div className="flex flex-wrap gap-4 text-sm">
                <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800/50 rounded-lg border border-gray-700/50">
                  <Clock className="w-4 h-4 text-primary-400" />
                  <span className="text-gray-300">{metadata.duration.toFixed(1)}s</span>
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800/50 rounded-lg border border-gray-700/50">
                  <Eye className="w-4 h-4 text-primary-400" />
                  <span className="text-gray-300">{metadata.width}x{metadata.height}</span>
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800/50 rounded-lg border border-gray-700/50">
                  <span className="text-gray-300">{metadata.fps.toFixed(1)} FPS</span>
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800/50 rounded-lg border border-gray-700/50">
                  <span className="text-gray-300">{metadata.frame_count} frames</span>
                </div>
              </div>
            </div>
            <div>
              <span
                className={`px-4 py-2 rounded-lg text-sm font-semibold shadow-lg ${
                  metadata.status === 'ready'
                    ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white'
                    : metadata.status === 'processing'
                    ? 'bg-gradient-to-r from-yellow-500 to-orange-500 text-white'
                    : metadata.status === 'failed'
                    ? 'bg-gradient-to-r from-red-500 to-rose-500 text-white'
                    : 'bg-gray-500/20 text-gray-400'
                }`}
              >
                {metadata.status}
              </span>
            </div>
          </div>
        </div>

        {isProcessing && (
          <div className="bg-gradient-to-r from-yellow-500/10 to-orange-500/10 border border-yellow-500/30 rounded-xl p-4 backdrop-blur-sm mt-4">
            <div className="flex items-center gap-3">
              <Loader2 className="w-5 h-5 text-yellow-400 animate-spin flex-shrink-0" />
              <div>
                <p className="text-yellow-400 font-medium">
                  Processing video with AI models...
                </p>
                <p className="text-yellow-400/70 text-sm mt-1">
                  Frame analysis and caption generation in progress. This may take a few minutes.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="grid lg:grid-cols-3 gap-6 mt-8">
        {/* Video Player */}
        <div className="lg:col-span-2 space-y-6">
          <VideoPlayer videoId={videoId} metadata={metadata} />
        </div>

        {/* Analysis Panel */}
        <div className="space-y-6">
          {/* Tabs */}
          <div className="bg-gray-900/80 backdrop-blur-sm rounded-xl shadow-xl border border-gray-700/50 p-2">
            <div className="flex flex-col space-y-1">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  disabled={isProcessing && tab.id !== 'query'}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                    activeTab === tab.id
                      ? 'bg-gradient-to-r from-primary-600 to-purple-600 text-white shadow-lg transform scale-[1.02]'
                      : 'text-gray-400 hover:bg-gray-800/50 hover:text-gray-200'
                  } disabled:opacity-40 disabled:cursor-not-allowed`}
                >
                  <tab.icon className="w-5 h-5 flex-shrink-0" />
                  <span className="font-medium">{tab.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Active Panel */}
          <div className="bg-gray-900/80 backdrop-blur-sm rounded-xl shadow-xl border border-gray-700/50">
            {activeTab === 'query' && <QueryPanel videoId={videoId} />}
            {activeTab === 'search' && <SearchPanel videoId={videoId} />}
            {activeTab === 'detect' && <DetectionPanel videoId={videoId} />}
            {activeTab === 'timeline' && <TimelinePanel videoId={videoId} />}
            {activeTab === 'redact' && <RedactionPanel videoId={videoId} />}
          </div>
        </div>
      </div>
    </div>
  );
}

