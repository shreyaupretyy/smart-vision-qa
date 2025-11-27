import whisper
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import logging
import tempfile
import os

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Audio transcription service using OpenAI Whisper"""
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize audio transcriber
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            output_path: Output audio file path (optional)
            
        Returns:
            Path to extracted audio file
        """
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "extracted_audio.wav")
        
        try:
            # Use ffmpeg to extract audio
            command = [
                "ffmpeg",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM codec
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite
                output_path
            ]
            
            subprocess.run(command, check=True, capture_output=True)
            logger.info(f"Extracted audio to {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio: {e}")
            # If ffmpeg fails, try alternative method with moviepy (requires installation)
            try:
                import moviepy.editor as mp
                video = mp.VideoFileClip(video_path)
                video.audio.write_audiofile(output_path, fps=16000)
                video.close()
                logger.info(f"Extracted audio using moviepy to {output_path}")
                return output_path
            except Exception as e2:
                logger.error(f"Alternative audio extraction also failed: {e2}")
                raise
    
    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            task: 'transcribe' or 'translate' (to English)
            
        Returns:
            Transcription result dictionary
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        logger.info(f"Transcribing audio from {audio_path}")
        
        options = {
            "task": task,
            "verbose": False
        }
        
        if language:
            options["language"] = language
        
        result = self.model.transcribe(audio_path, **options)
        
        logger.info(f"Transcription complete. Language: {result['language']}")
        return result
    
    def transcribe_video(
        self,
        video_path: str,
        language: Optional[str] = None,
        cleanup: bool = True
    ) -> Dict:
        """
        Transcribe audio from video file
        
        Args:
            video_path: Path to video file
            language: Language code or None for auto-detect
            cleanup: Whether to delete extracted audio file
            
        Returns:
            Transcription result dictionary
        """
        # Extract audio
        audio_path = self.extract_audio(video_path)
        
        try:
            # Transcribe
            result = self.transcribe_audio(audio_path, language)
            
            # Format segments
            formatted_segments = []
            for segment in result["segments"]:
                formatted_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "confidence": segment.get("no_speech_prob", 0.0)
                })
            
            return {
                "language": result["language"],
                "text": result["text"],
                "segments": formatted_segments
            }
            
        finally:
            # Cleanup
            if cleanup and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Cleaned up audio file: {audio_path}")
    
    def get_transcript_at_time(
        self,
        segments: List[Dict],
        timestamp: float,
        context_window: float = 5.0
    ) -> str:
        """
        Get transcript text around a specific timestamp
        
        Args:
            segments: List of transcript segments
            timestamp: Target timestamp in seconds
            context_window: Window size around timestamp in seconds
            
        Returns:
            Transcript text in the time window
        """
        start_time = max(0, timestamp - context_window)
        end_time = timestamp + context_window
        
        relevant_segments = [
            seg for seg in segments
            if seg["start"] <= end_time and seg["end"] >= start_time
        ]
        
        text = " ".join(seg["text"] for seg in relevant_segments)
        return text
    
    def search_transcript(
        self,
        segments: List[Dict],
        query: str,
        case_sensitive: bool = False
    ) -> List[Dict]:
        """
        Search for keywords in transcript
        
        Args:
            segments: List of transcript segments
            query: Search query
            case_sensitive: Whether search is case-sensitive
            
        Returns:
            List of matching segments with timestamps
        """
        if not case_sensitive:
            query = query.lower()
        
        matches = []
        for segment in segments:
            text = segment["text"] if case_sensitive else segment["text"].lower()
            
            if query in text:
                matches.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "match": query
                })
        
        logger.info(f"Found {len(matches)} matches for '{query}'")
        return matches
    
    def generate_subtitle_file(
        self,
        segments: List[Dict],
        output_path: str,
        format: str = "srt"
    ) -> str:
        """
        Generate subtitle file from segments
        
        Args:
            segments: List of transcript segments
            output_path: Output file path
            format: Subtitle format ('srt' or 'vtt')
            
        Returns:
            Path to subtitle file
        """
        if format.lower() == "srt":
            return self._generate_srt(segments, output_path)
        elif format.lower() == "vtt":
            return self._generate_vtt(segments, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_srt(self, segments: List[Dict], output_path: str) -> str:
        """Generate SRT subtitle file"""
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                start = self._format_timestamp(segment["start"])
                end = self._format_timestamp(segment["end"])
                text = segment["text"].strip()
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
        
        logger.info(f"Generated SRT file: {output_path}")
        return output_path
    
    def _generate_vtt(self, segments: List[Dict], output_path: str) -> str:
        """Generate WebVTT subtitle file"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            
            for segment in segments:
                start = self._format_timestamp(segment["start"], vtt=True)
                end = self._format_timestamp(segment["end"], vtt=True)
                text = segment["text"].strip()
                
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
        
        logger.info(f"Generated VTT file: {output_path}")
        return output_path
    
    def _format_timestamp(self, seconds: float, vtt: bool = False) -> str:
        """Format timestamp for subtitle files"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        if vtt:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
        else:
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def get_transcript_summary(self, segments: List[Dict]) -> Dict:
        """
        Get summary statistics of transcript
        
        Args:
            segments: List of transcript segments
            
        Returns:
            Summary dictionary
        """
        if not segments:
            return {
                "total_segments": 0,
                "total_duration": 0.0,
                "word_count": 0,
                "avg_confidence": 0.0
            }
        
        total_duration = segments[-1]["end"] if segments else 0.0
        full_text = " ".join(seg["text"] for seg in segments)
        word_count = len(full_text.split())
        
        confidences = [seg.get("confidence", 1.0) for seg in segments]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "total_segments": len(segments),
            "total_duration": total_duration,
            "word_count": word_count,
            "avg_confidence": 1.0 - avg_confidence,  # Convert no_speech_prob to confidence
            "characters": len(full_text)
        }
