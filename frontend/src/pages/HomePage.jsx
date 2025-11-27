import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Video, Brain, Search, Shield, Users, Zap } from 'lucide-react';
import VideoUpload from '../components/VideoUpload';
import { motion } from 'framer-motion';

export default function HomePage() {
  const navigate = useNavigate();

  const handleUploadComplete = (result) => {
    // Navigate to analyze page
    navigate(`/analyze/${result.video_id}`);
  };

  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Q&A',
      description: 'Ask natural language questions about your videos and get instant, accurate answers.',
    },
    {
      icon: Search,
      title: 'Semantic Search',
      description: 'Find specific moments in videos using natural language descriptions.',
    },
    {
      icon: Video,
      title: 'Object Detection',
      description: 'Real-time detection and tracking of objects across video frames.',
    },
    {
      icon: Shield,
      title: 'Privacy Protection',
      description: 'Automatically redact faces and sensitive objects for privacy compliance.',
    },
    {
      icon: Users,
      title: 'Collaboration',
      description: 'Real-time collaborative annotation and analysis with your team.',
    },
    {
      icon: Zap,
      title: 'Event Timeline',
      description: 'AI-generated timeline of key events with smart clustering.',
    },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center mb-16"
      >
        <h1 className="text-5xl md:text-6xl font-bold text-white mb-6">
          AI-Powered Video
          <span className="bg-gradient-to-r from-primary-400 to-blue-500 bg-clip-text text-transparent">
            {' '}Understanding
          </span>
        </h1>
        <p className="text-xl text-gray-400 max-w-3xl mx-auto mb-8">
          Upload videos and ask questions, search for specific moments, detect objects,
          and generate insights—all powered by state-of-the-art AI models.
        </p>
        <div className="flex justify-center gap-4">
          <a href="#upload" className="btn-primary text-lg px-8 py-3">
            Get Started
          </a>
          <a
            href="https://github.com/shreyaupretyy/smart-vision-qa"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-secondary text-lg px-8 py-3"
          >
            View on GitHub
          </a>
        </div>
      </motion.div>

      {/* Features Grid */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-16"
      >
        {features.map((feature, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 * index }}
            className="card hover:border-primary-500 transition-colors cursor-pointer group"
          >
            <div className="p-3 bg-primary-600/10 rounded-lg w-fit mb-4 group-hover:bg-primary-600/20 transition-colors">
              <feature.icon className="w-6 h-6 text-primary-400" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
            <p className="text-gray-400">{feature.description}</p>
          </motion.div>
        ))}
      </motion.div>

      {/* Upload Section */}
      <motion.div
        id="upload"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
        className="max-w-2xl mx-auto"
      >
        <VideoUpload onUploadComplete={handleUploadComplete} />
      </motion.div>

      {/* Stats Section */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.6 }}
        className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-8"
      >
        {[
          { label: 'AI Models', value: '4+' },
          { label: 'Object Classes', value: '80+' },
          { label: 'Languages', value: '99+' },
          { label: 'Real-time', value: '100%' },
        ].map((stat, index) => (
          <div key={index} className="text-center">
            <div className="text-4xl font-bold text-primary-400 mb-2">{stat.value}</div>
            <div className="text-gray-400">{stat.label}</div>
          </div>
        ))}
      </motion.div>
    </div>
  );
}
