import { useState } from 'react';
import { Search as SearchIcon, Loader2 } from 'lucide-react';
import { searchVideo } from '../services/api';
import toast from 'react-hot-toast';

export default function SearchPanel({ videoId }) {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResults([]);

    try {
      const data = await searchVideo(videoId, query, 5);
      setResults(data.results);
    } catch (error) {
      console.error('Search error:', error);
      toast.error(error.response?.data?.detail || 'Failed to search video');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-white">Semantic Search</h3>
      
      <form onSubmit={handleSearch} className="space-y-3">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g., person wearing red shirt"
          className="input-field"
          disabled={loading}
        />
        
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Searching...
            </>
          ) : (
            <>
              <SearchIcon className="w-4 h-4" />
              Search
            </>
          )}
        </button>
      </form>

      {results.length > 0 && (
        <div className="space-y-3">
          <p className="text-sm text-gray-400">{results.length} results found</p>
          
          {results.map((result, index) => (
            <div key={index} className="bg-gray-700/50 rounded-lg p-3 space-y-2">
              <div className="flex justify-between items-start">
                <span className="text-white font-medium">Frame #{result.frame_number}</span>
                <span className="text-sm text-primary-400">
                  {(result.similarity * 100).toFixed(1)}% match
                </span>
              </div>
              
              <p className="text-sm text-gray-400">
                Time: {result.timestamp.toFixed(2)}s
              </p>
              
              {result.description && (
                <p className="text-sm text-gray-300">{result.description}</p>
              )}
            </div>
          ))}
        </div>
      )}

      {!loading && results.length === 0 && query && (
        <p className="text-center text-gray-400 py-4">No results found</p>
      )}
    </div>
  );
}
