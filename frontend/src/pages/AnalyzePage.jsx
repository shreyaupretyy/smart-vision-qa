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
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="w-8 h-8 text-primary-400 animate-spin" />
      </div>
    );
  }

  if (!metadata) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-xl text-gray-400">Video not found</p>
        </div>
      </div>
    );
  }

  const isProcessing = metadata.status === 'processing' || metadata.status === 'uploading';

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Video Info */}
      <div className="card mb-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-white mb-2">{metadata.filename}</h1>
            <div className="flex flex-wrap gap-4 text-sm text-gray-400">
              <span>Duration: {metadata.duration.toFixed(1)}s</span>
              <span>Resolution: {metadata.width}x{metadata.height}</span>
              <span>FPS: {metadata.fps.toFixed(1)}</span>
              <span>Frames: {metadata.frame_count}</span>
            </div>
          </div>
          <div>
            <span
              className={`px-3 py-1 rounded-full text-sm font-medium ${
                metadata.status === 'ready'
                  ? 'bg-green-500/20 text-green-400'
                  : metadata.status === 'processing'
                  ? 'bg-yellow-500/20 text-yellow-400'
                  : metadata.status === 'failed'
                  ? 'bg-red-500/20 text-red-400'
                  : 'bg-gray-500/20 text-gray-400'
              }`}
            >
              {metadata.status}
            </span>
          </div>
        </div>

        {isProcessing && (
          <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4">
            <div className="flex items-center gap-3">
              <Loader2 className="w-5 h-5 text-yellow-400 animate-spin" />
              <p className="text-yellow-400">
                Video is being processed. Some features may not be available yet.
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Video Player */}
        <div className="lg:col-span-2">
          <VideoPlayer videoId={videoId} metadata={metadata} />
        </div>

        {/* Analysis Panel */}
        <div className="space-y-6">
          {/* Tabs */}
          <div className="card">
            <div className="flex flex-col space-y-2">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  disabled={isProcessing && tab.id !== 'query'}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                    activeTab === tab.id
                      ? 'bg-primary-600 text-white'
                      : 'text-gray-400 hover:bg-gray-700'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  <tab.icon className="w-5 h-5" />
                  <span className="font-medium">{tab.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Active Panel */}
          <div className="card">
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
