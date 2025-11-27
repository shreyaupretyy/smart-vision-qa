"""
Performance metrics and model evaluation service.
Tracks accuracy, latency, and model performance for continuous improvement.
"""
import time
import logging
import psutil
import torch
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from backend.core.database import VideoMetrics, ModelComparison, Query

logger = logging.getLogger(__name__)


class MetricsService:
    """Service for tracking and analyzing model performance"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def track_processing_time(self, operation_name: str):
        """Context manager for tracking processing time"""
        class Timer:
            def __init__(self, name):
                self.name = name
                self.start_time = None
                self.duration = None
                
            def __enter__(self):
                self.start_time = time.time()
                return self
                
            def __exit__(self, *args):
                self.duration = time.time() - self.start_time
                logger.info(f"⏱️  {self.name}: {self.duration:.2f}s")
                
        return Timer(operation_name)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_peak_memory_delta(self) -> float:
        """Get peak memory increase from initial"""
        return self.get_memory_usage() - self.initial_memory
    
    def create_video_metrics(
        self,
        db: Session,
        video_id: str,
        frame_extraction_time: float,
        caption_generation_time: float,
        embedding_generation_time: float,
        total_processing_time: float,
        captions: List[str],
        gpu_used: bool = False
    ) -> VideoMetrics:
        """Create metrics record for video processing"""
        
        metrics = VideoMetrics(
            video_id=video_id,
            frame_extraction_time=frame_extraction_time,
            caption_generation_time=caption_generation_time,
            embedding_generation_time=embedding_generation_time,
            total_processing_time=total_processing_time,
            average_caption_length=sum(len(c.split()) for c in captions) / len(captions) if captions else 0,
            peak_memory_usage=self.get_peak_memory_delta(),
            gpu_used=gpu_used
        )
        
        db.add(metrics)
        db.commit()
        db.refresh(metrics)
        
        logger.info(f"✓ Created metrics for video {video_id}")
        return metrics
    
    def update_query_metrics(
        self,
        db: Session,
        video_id: str,
        query_time: float,
        confidence: float
    ):
        """Update video metrics with new query data"""
        metrics = db.query(VideoMetrics).filter(VideoMetrics.video_id == video_id).first()
        
        if metrics:
            # Update running averages
            total_queries = metrics.total_queries + 1
            
            if metrics.average_query_time:
                new_avg_time = (metrics.average_query_time * metrics.total_queries + query_time) / total_queries
            else:
                new_avg_time = query_time
                
            if metrics.average_confidence:
                new_avg_conf = (metrics.average_confidence * metrics.total_queries + confidence) / total_queries
            else:
                new_avg_conf = confidence
            
            metrics.total_queries = total_queries
            metrics.average_query_time = new_avg_time
            metrics.average_confidence = new_avg_conf
            metrics.updated_at = datetime.utcnow()
            
            db.commit()
    
    def log_model_comparison(
        self,
        db: Session,
        model_name: str,
        model_type: str,
        query: str,
        answer: str,
        confidence: float,
        processing_time: float,
        accuracy_score: Optional[float] = None
    ):
        """Log model performance for comparison"""
        comparison = ModelComparison(
            model_name=model_name,
            model_type=model_type,
            query=query,
            answer=answer,
            confidence=confidence,
            processing_time=processing_time,
            accuracy_score=accuracy_score
        )
        
        db.add(comparison)
        db.commit()
    
    def get_video_statistics(self, db: Session, video_id: str) -> Dict:
        """Get comprehensive statistics for a video"""
        metrics = db.query(VideoMetrics).filter(VideoMetrics.video_id == video_id).first()
        queries = db.query(Query).filter(Query.video_id == video_id).all()
        
        if not metrics:
            return {}
        
        return {
            "processing": {
                "frame_extraction_time": metrics.frame_extraction_time,
                "caption_generation_time": metrics.caption_generation_time,
                "embedding_generation_time": metrics.embedding_generation_time,
                "total_processing_time": metrics.total_processing_time,
                "gpu_used": metrics.gpu_used,
                "peak_memory_mb": metrics.peak_memory_usage
            },
            "quality": {
                "average_caption_length": metrics.average_caption_length,
                "unique_objects_detected": metrics.unique_objects_detected,
                "total_queries": metrics.total_queries,
                "average_query_time": metrics.average_query_time,
                "average_confidence": metrics.average_confidence
            },
            "queries": [
                {
                    "question": q.question,
                    "answer": q.answer,
                    "confidence": q.confidence,
                    "processing_time": q.processing_time,
                    "model": q.model_used,
                    "timestamp": q.created_at.isoformat()
                }
                for q in queries
            ]
        }
    
    def get_model_comparison_stats(self, db: Session, model_type: Optional[str] = None) -> Dict:
        """Get comparative statistics across models"""
        query = db.query(ModelComparison)
        if model_type:
            query = query.filter(ModelComparison.model_type == model_type)
        
        comparisons = query.all()
        
        if not comparisons:
            return {}
        
        # Group by model
        model_stats = {}
        for comp in comparisons:
            if comp.model_name not in model_stats:
                model_stats[comp.model_name] = {
                    "count": 0,
                    "total_time": 0,
                    "total_confidence": 0,
                    "accuracies": []
                }
            
            stats = model_stats[comp.model_name]
            stats["count"] += 1
            stats["total_time"] += comp.processing_time
            stats["total_confidence"] += comp.confidence
            if comp.accuracy_score is not None:
                stats["accuracies"].append(comp.accuracy_score)
        
        # Calculate averages
        result = {}
        for model, stats in model_stats.items():
            result[model] = {
                "total_queries": stats["count"],
                "avg_processing_time": stats["total_time"] / stats["count"],
                "avg_confidence": stats["total_confidence"] / stats["count"],
                "avg_accuracy": sum(stats["accuracies"]) / len(stats["accuracies"]) if stats["accuracies"] else None
            }
        
        return result
    
    def evaluate_answer_quality(self, answer: str, ground_truth: Optional[str] = None) -> Dict:
        """
        Evaluate answer quality with various metrics.
        If ground truth is provided, calculates accuracy metrics.
        """
        metrics = {
            "length": len(answer.split()),
            "has_context": len(answer.split()) > 5,
            "completeness_score": min(len(answer.split()) / 10, 1.0)  # Normalized to 1.0
        }
        
        if ground_truth:
            # Simple word overlap metric
            answer_words = set(answer.lower().split())
            truth_words = set(ground_truth.lower().split())
            
            if truth_words:
                overlap = len(answer_words & truth_words) / len(truth_words)
                metrics["word_overlap"] = overlap
                metrics["exact_match"] = answer.lower().strip() == ground_truth.lower().strip()
        
        return metrics
    
    def benchmark_models(
        self,
        models: List[Tuple[str, callable]],
        test_data: List[Tuple],
        model_type: str,
        db: Session
    ) -> Dict:
        """
        Benchmark multiple models against test data.
        
        Args:
            models: List of (model_name, model_function) tuples
            test_data: List of (input, expected_output) tuples
            model_type: Type of model being tested
            db: Database session
        
        Returns:
            Benchmark results
        """
        results = {}
        
        for model_name, model_fn in models:
            logger.info(f"Benchmarking {model_name}...")
            
            timings = []
            accuracies = []
            
            for input_data, expected in test_data:
                start = time.time()
                output = model_fn(input_data)
                duration = time.time() - start
                
                timings.append(duration)
                
                # Calculate accuracy if expected output provided
                if expected:
                    quality = self.evaluate_answer_quality(str(output), str(expected))
                    if "word_overlap" in quality:
                        accuracies.append(quality["word_overlap"])
                
                # Log to database
                self.log_model_comparison(
                    db=db,
                    model_name=model_name,
                    model_type=model_type,
                    query=str(input_data),
                    answer=str(output),
                    confidence=0.85,
                    processing_time=duration,
                    accuracy_score=accuracies[-1] if accuracies else None
                )
            
            results[model_name] = {
                "avg_time": sum(timings) / len(timings),
                "min_time": min(timings),
                "max_time": max(timings),
                "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
                "total_tests": len(test_data)
            }
        
        return results


# Global metrics service instance
metrics_service = MetricsService()
