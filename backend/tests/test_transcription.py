"""
Unit tests for transcription service.
"""
import pytest
from backend.services.transcription import AudioTranscriber

@pytest.fixture
def audio_transcriber():
    """Create AudioTranscriber instance."""
    return AudioTranscriber()

def test_transcriber_initialization(audio_transcriber):
    """Test AudioTranscriber initialization."""
    assert audio_transcriber is not None
    assert hasattr(audio_transcriber, 'transcribe_video')

def test_transcription_structure(mock_transcription):
    """Test transcription result structure."""
    assert 'text' in mock_transcription
    assert 'segments' in mock_transcription
    assert 'language' in mock_transcription
    assert isinstance(mock_transcription['segments'], list)

def test_transcription_segments(mock_transcription):
    """Test transcription segment structure."""
    for segment in mock_transcription['segments']:
        assert 'start' in segment
        assert 'end' in segment
        assert 'text' in segment
        assert segment['start'] < segment['end']

def test_transcription_timing(mock_transcription):
    """Test transcription timing is sequential."""
    segments = mock_transcription['segments']
    for i in range(len(segments) - 1):
        assert segments[i]['end'] <= segments[i + 1]['start']

def test_language_code_format(mock_transcription):
    """Test language code format."""
    lang = mock_transcription['language']
    assert isinstance(lang, str)
    assert len(lang) == 2  # ISO 639-1 code
