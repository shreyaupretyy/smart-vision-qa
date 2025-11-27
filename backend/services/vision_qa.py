from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import torch
from PIL import Image
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VisionQA:
    """Vision-Language Question Answering service using BLIP"""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize Vision QA service
        
        Args:
            model_name: Hugging Face model name
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.caption_processor = None
        self.caption_model = None
        self.qa_processor = None
        self.qa_model = None
        self._load_models()
    
    def _load_models(self):
        """Load BLIP models"""
        try:
            # Load captioning model
            logger.info(f"Loading caption model: {self.model_name}")
            self.caption_processor = BlipProcessor.from_pretrained(self.model_name)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                self.model_name
            ).to(self.device)
            
            # Load QA model
            qa_model_name = "Salesforce/blip-vqa-base"
            logger.info(f"Loading QA model: {qa_model_name}")
            self.qa_processor = BlipProcessor.from_pretrained(qa_model_name)
            self.qa_model = BlipForQuestionAnswering.from_pretrained(
                qa_model_name
            ).to(self.device)
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def generate_caption(self, image: np.ndarray, max_length: int = 50) -> str:
        """
        Generate caption for an image
        
        Args:
            image: Input image as numpy array (BGR format)
            max_length: Maximum caption length
            
        Returns:
            Generated caption
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image)
        
        inputs = self.caption_processor(pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.caption_model.generate(**inputs, max_length=max_length)
        
        caption = self.caption_processor.decode(output[0], skip_special_tokens=True)
        return caption
    
    def answer_question(
        self, 
        image: np.ndarray, 
        question: str,
        max_length: int = 50,
        include_context: bool = True
    ) -> Tuple[str, float]:
        """
        Answer a question about an image
        
        Args:
            image: Input image as numpy array (BGR format)
            question: Question to answer
            max_length: Maximum answer length
            include_context: Whether to include scene context for more detailed answers
            
        Returns:
            Tuple of (answer, confidence)
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image)
        
        inputs = self.qa_processor(
            pil_image, 
            question, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output = self.qa_model.generate(**inputs, max_length=max_length)
        
        answer = self.qa_processor.decode(output[0], skip_special_tokens=True)
        
        # Generate more detailed answer by including scene caption
        if include_context and len(answer.split()) <= 3:
            try:
                caption = self.generate_caption(image, max_length=75)
                # Create a more natural response
                question_lower = question.lower()
                
                if any(word in question_lower for word in ["what is", "what's", "describe", "tell me about", "what am i seeing"]):
                    answer = f"{caption}. {answer.capitalize()}"
                elif "who" in question_lower:
                    answer = f"In this scene, {caption.lower()}. {answer.capitalize()}"
                elif "where" in question_lower:
                    answer = f"The scene shows {caption.lower()}. Location: {answer}"
                elif "when" in question_lower:
                    answer = f"{caption}. {answer.capitalize()}"
                elif "why" in question_lower or "how" in question_lower:
                    answer = f"Based on the scene ({caption.lower()}), {answer.lower()}"
                else:
                    answer = f"{caption}. To answer your question: {answer.lower()}"
            except Exception as e:
                logger.warning(f"Failed to add context to answer: {e}")
        
        # Get confidence score (simplified)
        confidence = 0.85  # Placeholder - BLIP doesn't directly provide confidence
        
        return answer, confidence
    
    def analyze_frames(
        self,
        frames: List[Tuple[int, float, np.ndarray]],
        generate_captions: bool = True
    ) -> List[Dict]:
        """
        Analyze multiple frames
        
        Args:
            frames: List of (frame_number, timestamp, frame_array) tuples
            generate_captions: Whether to generate captions
            
        Returns:
            List of analysis results
        """
        results = []
        
        for frame_num, timestamp, frame in frames:
            result = {
                "frame_number": frame_num,
                "timestamp": timestamp,
            }
            
            if generate_captions:
                caption = self.generate_caption(frame)
                result["caption"] = caption
            
            results.append(result)
        
        logger.info(f"Analyzed {len(results)} frames")
        return results
    
    def query_video(
        self,
        video_path: str,
        question: str,
        sample_rate: int = 5,
        top_k: int = 8
    ) -> Dict:
        """
        Query entire video with a question using improved multi-frame reasoning
        
        Args:
            video_path: Path to video file
            question: Question to answer
            sample_rate: Sample 1 frame every N seconds (increased to 5 for faster processing)
            top_k: Return top K most relevant frames
            
        Returns:
            Dictionary with comprehensive answer and relevant frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        frame_results = []
        frame_interval = int(fps * sample_rate)
        
        # Limit maximum frames to analyze for performance
        max_frames_to_analyze = min(100, total_frames // frame_interval)
        
        logger.info(f"Querying video with question: '{question}'")
        logger.info(f"Analyzing {max_frames_to_analyze} frames from {duration:.1f}s video")
        
        # Sample frames evenly across video
        frame_indices = [int(i * total_frames / max_frames_to_analyze) 
                        for i in range(max_frames_to_analyze)]
        
        for frame_num in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            timestamp = frame_num / fps
            
            # First get caption for scene understanding
            caption = self.generate_caption(frame, max_length=75)
            
            # Get answer with full context
            answer, confidence = self.answer_question(frame, question, include_context=True)
            
            # Calculate better confidence based on answer relevance
            answer_length = len(answer.split())
            caption_relevance = self._calculate_relevance(question, caption)
            # Combine factors: longer answers usually better, caption relevance matters
            adjusted_confidence = (0.4 * min(1.0, answer_length / 15) + 
                                 0.6 * caption_relevance)
            
            frame_results.append({
                "frame_number": frame_num,
                "timestamp": timestamp,
                "answer": answer,
                "caption": caption,
                "confidence": adjusted_confidence
            })
        
        cap.release()
        
        if not frame_results:
            return {
                "question": question,
                "answer": "Unable to analyze video frames",
                "confidence": 0.0,
                "relevant_frames": [],
                "detailed_results": []
            }
        
        # Sort by confidence and get top K
        frame_results.sort(key=lambda x: x["confidence"], reverse=True)
        top_results = frame_results[:top_k]
        
        # Synthesize comprehensive answer from multiple frames
        final_answer = self._synthesize_multi_frame_answer(
            question, 
            top_results,
            total_frames / fps  # video duration
        )
        
        return {
            "question": question,
            "answer": final_answer,
            "confidence": top_results[0]["confidence"] if top_results else 0.0,
            "relevant_frames": [r["frame_number"] for r in top_results],
            "detailed_results": top_results
        }
    
    def _synthesize_multi_frame_answer(
        self,
        question: str,
        top_results: List[Dict],
        video_duration: float
    ) -> str:
        """
        Synthesize a comprehensive answer from multiple frames
        
        Args:
            question: Original question
            top_results: Top matching frames with answers
            video_duration: Total video duration in seconds
            
        Returns:
            Synthesized comprehensive answer
        """
        if not top_results:
            return "No relevant information found in the video."
        
        # Extract answers and captions
        answers = [r["answer"] for r in top_results]
        captions = [r["caption"] for r in top_results]
        
        # Get the most detailed answer (longest)
        best_answer = max(answers, key=len)
        
        # Check if answers are consistent
        unique_answers = []
        for ans in answers:
            # Extract key words (simple approach)
            ans_lower = ans.lower()
            if not any(ans_lower in existing.lower() or existing.lower() in ans_lower 
                      for existing in unique_answers):
                unique_answers.append(ans)
        
        # Build comprehensive response
        if len(unique_answers) == 1:
            # Consistent answer across frames - add temporal context
            timestamps = [r["timestamp"] for r in top_results]
            if len(timestamps) > 1:
                time_range = f"throughout the video (at {timestamps[0]:.1f}s to {timestamps[-1]:.1f}s)"
                final_answer = f"{best_answer}\n\nThis is observed {time_range}."
            else:
                final_answer = f"{best_answer}\n\nThis occurs at {timestamps[0]:.1f}s in the video."
        else:
            # Different observations at different times
            question_lower = question.lower()
            
            if any(word in question_lower for word in ["what happens", "what is happening", "describe", "summary"]):
                # Build narrative from captions
                temporal_descriptions = []
                for i, result in enumerate(top_results[:3]):
                    temporal_descriptions.append(
                        f"At {result['timestamp']:.1f}s: {result['caption']}"
                    )
                final_answer = f"{best_answer}\n\nTemporal breakdown:\n" + "\n".join(temporal_descriptions)
            
            elif "how many" in question_lower or "count" in question_lower:
                # For counting questions, try to aggregate
                final_answer = f"{best_answer}\n\nNote: The count may vary across different moments in the video."
            
            else:
                # General case: provide the best answer with context
                final_answer = f"{best_answer}\n\nMultiple observations found across the video, this represents the most relevant answer."
        
        return final_answer
    
    def describe_scene(self, image: np.ndarray) -> Dict:
        """
        Generate detailed scene description
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with scene description
        """
        caption = self.generate_caption(image)
        
        # Generate additional details with questions
        questions = [
            "How many people are in the image?",
            "What is the main activity?",
            "What objects are visible?",
            "What is the setting?"
        ]
        
        details = {}
        for question in questions:
            answer, _ = self.answer_question(image, question)
            details[question] = answer
        
        return {
            "caption": caption,
            "details": details
        }
    
    def _calculate_relevance(self, question: str, text: str) -> float:
        """
        Calculate simple relevance score between question and text
        
        Args:
            question: Question text
            text: Text to compare (caption/answer)
            
        Returns:
            Relevance score between 0 and 1
        """
        # Simple keyword matching for relevance
        question_words = set(question.lower().split())
        text_words = set(text.lower().split())
        
        # Remove common stop words
        stop_words = {"a", "an", "the", "is", "are", "was", "were", "this", "that", 
                     "what", "how", "when", "where", "who", "which", "do", "does"}
        question_words -= stop_words
        text_words -= stop_words
        
        if not question_words:
            return 0.5
        
        # Calculate overlap
        overlap = len(question_words & text_words)
        relevance = overlap / len(question_words)
        
        return min(1.0, relevance + 0.3)  # Boost baseline relevance
    
    def compare_frames(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> Dict:
        """
        Compare two frames and describe changes
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Dictionary with comparison results
        """
        caption1 = self.generate_caption(frame1)
        caption2 = self.generate_caption(frame2)
        
        # Ask about differences
        question = "What has changed?"
        answer, confidence = self.answer_question(frame2, question)
        
        return {
            "frame1_caption": caption1,
            "frame2_caption": caption2,
            "changes": answer,
            "confidence": confidence
        }
    
    def batch_caption_frames(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8
    ) -> List[str]:
        """
        Generate captions for multiple frames in batches
        
        Args:
            frames: List of frames
            batch_size: Batch size for processing
            
        Returns:
            List of captions
        """
        captions = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            
            for frame in batch:
                caption = self.generate_caption(frame)
                captions.append(caption)
        
        return captions
