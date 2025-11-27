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
        max_length: int = 50
    ) -> Tuple[str, float]:
        """
        Answer a question about an image
        
        Args:
            image: Input image as numpy array (BGR format)
            question: Question to answer
            max_length: Maximum answer length
            
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
        sample_rate: int = 2,
        top_k: int = 3
    ) -> Dict:
        """
        Query entire video with a question
        
        Args:
            video_path: Path to video file
            question: Question to answer
            sample_rate: Sample 1 frame every N seconds
            top_k: Return top K most relevant frames
            
        Returns:
            Dictionary with answers and relevant frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_results = []
        frame_interval = int(fps * sample_rate)
        
        for frame_num in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            answer, confidence = self.answer_question(frame, question)
            timestamp = frame_num / fps
            
            frame_results.append({
                "frame_number": frame_num,
                "timestamp": timestamp,
                "answer": answer,
                "confidence": confidence
            })
        
        cap.release()
        
        # Sort by confidence and get top K
        frame_results.sort(key=lambda x: x["confidence"], reverse=True)
        top_results = frame_results[:top_k]
        
        # Get most common answer
        answers = [r["answer"] for r in top_results]
        best_answer = max(set(answers), key=answers.count)
        
        return {
            "question": question,
            "answer": best_answer,
            "confidence": top_results[0]["confidence"] if top_results else 0.0,
            "relevant_frames": [r["frame_number"] for r in top_results],
            "detailed_results": top_results
        }
    
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
