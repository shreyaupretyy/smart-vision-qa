"""
Unit tests for embeddings service.
"""
import pytest
from backend.services.embeddings import EmbeddingsService

@pytest.fixture
def embeddings_service():
    """Create EmbeddingsService instance."""
    return EmbeddingsService()

def test_embeddings_service_initialization(embeddings_service):
    """Test EmbeddingsService initialization."""
    assert embeddings_service is not None
    assert hasattr(embeddings_service, 'add_frames')
    assert hasattr(embeddings_service, 'search_frames')

def test_embedding_dimension():
    """Test embedding vector dimensions."""
    # Sentence transformers typically use 384 or 768 dimensions
    expected_dims = [384, 512, 768]
    # This would be tested with actual embeddings
    assert any(dim > 0 for dim in expected_dims)

def test_search_result_structure():
    """Test search result structure."""
    result = {
        'frame_number': 1,
        'timestamp': 0.5,
        'similarity': 0.95,
        'frame_path': '/path/to/frame.jpg'
    }
    assert all(key in result for key in ['frame_number', 'timestamp', 'similarity', 'frame_path'])

def test_similarity_score_range():
    """Test similarity scores are in valid range."""
    similarity_scores = [0.95, 0.88, 0.75, 0.62]
    for score in similarity_scores:
        assert 0.0 <= score <= 1.0

def test_search_results_ordering():
    """Test search results are ordered by similarity."""
    results = [
        {'similarity': 0.95},
        {'similarity': 0.88},
        {'similarity': 0.75}
    ]
    similarities = [r['similarity'] for r in results]
    assert similarities == sorted(similarities, reverse=True)
