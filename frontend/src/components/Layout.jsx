import { Video, Menu, X } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useState } from 'react';

export default function Layout({ children }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      {/* Header */}
      <header className="bg-gray-800/50 backdrop-blur-lg border-b border-gray-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center space-x-3 group">
              <div className="p-2 bg-primary-600 rounded-lg group-hover:bg-primary-500 transition-colors">
                <Video className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold text-white">SmartVisionQA</span>
            </Link>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex space-x-8">
              <Link
                to="/"
                className="text-gray-300 hover:text-white transition-colors"
              >
                Home
              </Link>
              <a
                href="/docs"
                className="text-gray-300 hover:text-white transition-colors"
              >
                Documentation
              </a>
              <a
                href="https://github.com/shreyaupretyy/smart-vision-qa"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-300 hover:text-white transition-colors"
              >
                GitHub
              </a>
            </nav>

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-lg text-gray-300 hover:bg-gray-700"
            >
              {mobileMenuOpen ? <X /> : <Menu />}
            </button>
          </div>

          {/* Mobile menu */}
          {mobileMenuOpen && (
            <div className="md:hidden py-4 space-y-2">
              <Link
                to="/"
                className="block px-4 py-2 text-gray-300 hover:bg-gray-700 rounded-lg"
                onClick={() => setMobileMenuOpen(false)}
              >
                Home
              </Link>
              <a
                href="/docs"
                className="block px-4 py-2 text-gray-300 hover:bg-gray-700 rounded-lg"
              >
                Documentation
              </a>
              <a
                href="https://github.com/shreyaupretyy/smart-vision-qa"
                target="_blank"
                rel="noopener noreferrer"
                className="block px-4 py-2 text-gray-300 hover:bg-gray-700 rounded-lg"
              >
                GitHub
              </a>
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="min-h-[calc(100vh-4rem)]">{children}</main>

      {/* Footer */}
      <footer className="bg-gray-800/50 border-t border-gray-700 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center text-gray-400">
            <p className="mb-2">
              SmartVisionQA - AI-Powered Video Understanding System
            </p>
            <p className="text-sm">
              Built with ❤️ using React, FastAPI, YOLOv8, BLIP, and Whisper
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
