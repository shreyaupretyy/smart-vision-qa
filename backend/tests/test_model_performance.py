"""
Comprehensive test suite for model performance and accuracy.
Tests vision QA, object detection, and embeddings services.
"""
import pytest
import numpy as np
import cv2
from PIL import Image
import time
import torch
from pathlib import Path

from backend.services.vision_qa import VisionQA
from backend.services.object_detection import ObjectDetector
from backend.services.advanced_embeddings import AdvancedEmbeddingsService
from backend.services.metrics import MetricsService


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Draw a simple shape
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
    cv2.circle(img, (100, 100), 30, (0, 255, 0), -1)
    return img


@pytest.fixture
def vision_qa_service():
    """Initialize Vision QA service"""
    return VisionQA()


@pytest.fixture
def object_detector():
    """Initialize Object Detector"""
    return ObjectDetector()


@pytest.fixture
def embeddings_service():
    """Initialize Advanced Embeddings service"""
    return AdvancedEmbeddingsService()


@pytest.fixture
def metrics_service():
    """Initialize Metrics service"""
    return MetricsService()


class TestVisionQA:
    """Test suite for Vision QA service"""
    
    def test_caption_generation(self, vision_qa_service, sample_image):
        """Test basic caption generation"""
        caption = vision_qa_service.generate_caption(sample_image)
        
        assert isinstance(caption, str)
        assert len(caption) > 0
        print(f"Generated caption: {caption}")
    
    def test_question_answering(self, vision_qa_service, sample_image):
        """Test question answering with context"""
        question = "What shapes are visible?"
        answer, confidence = vision_qa_service.answer_question(
            sample_image, 
            question,
            include_context=True
        )
        
        assert isinstance(answer, str)
        assert isinstance(confidence, float)
        assert len(answer) > 0
        assert 0 <= confidence <= 1
        print(f"Q: {question}\nA: {answer} (confidence: {confidence:.2f})")
    
    def test_answer_quality(self, vision_qa_service, sample_image):
        """Test that answers include context for short responses"""
        question = "What is this?"
        answer, _ = vision_qa_service.answer_question(
            sample_image,
            question,
            include_context=True
        )
        
        # Should have more than just a one-word answer
        word_count = len(answer.split())
        assert word_count > 3, f"Answer too short: '{answer}'"
    
    def test_performance_timing(self, vision_qa_service, sample_image, metrics_service):
        """Test inference speed"""
        with metrics_service.track_processing_time("Caption Generation") as timer:
            caption = vision_qa_service.generate_caption(sample_image)
        
        assert timer.duration < 5.0, f"Caption generation too slow: {timer.duration:.2f}s"
        
        with metrics_service.track_processing_time("Question Answering") as timer:
            answer, _ = vision_qa_service.answer_question(sample_image, "What is this?")
        
        assert timer.duration < 5.0, f"QA too slow: {timer.duration:.2f}s"
    
    def test_gpu_utilization(self, vision_qa_service):
        """Test that GPU is being used if available"""
        if torch.cuda.is_available():
            assert vision_qa_service.device == "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            assert vision_qa_service.device == "cpu"
            print("Running on CPU")


class TestObjectDetection:
    """Test suite for Object Detection service"""
    
    def test_basic_detection(self, object_detector, sample_image):
        """Test basic object detection"""
        detections = object_detector.detect_objects(sample_image)
        
        assert isinstance(detections, list)
        # Note: Our simple test image might not detect anything
        print(f"Detected {len(detections)} objects")
    
    def test_confidence_threshold(self, object_detector, sample_image):
        """Test confidence threshold filtering"""
        high_conf = object_detector.detect_objects(sample_image, confidence_threshold=0.8)
        low_conf = object_detector.detect_objects(sample_image, confidence_threshold=0.3)
        
        # Higher threshold should have fewer or equal detections
        assert len(high_conf) <= len(low_conf)
    
    def test_detection_format(self, object_detector, sample_image):
        """Test detection output format"""
        detections = object_detector.detect_objects(sample_image, confidence_threshold=0.1)
        
        for det in detections:
            assert "class" in det
            assert "confidence" in det
            assert "bbox" in det
            assert len(det["bbox"]) == 4
            assert all(isinstance(x, (int, float)) for x in det["bbox"])


class TestAdvancedEmbeddings:
    """Test suite for Advanced Embeddings service"""
    
    def test_visual_embedding(self, embeddings_service, sample_image):
        """Test visual embedding generation"""
        embedding = embeddings_service.create_visual_embedding(sample_image)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # 1D vector
        assert len(embedding) > 0
        print(f"Visual embedding dimension: {len(embedding)}")
    
    def test_text_embedding(self, embeddings_service):
        """Test text embedding generation"""
        text = "A person walking in the park"
        embedding = embeddings_service.create_text_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert len(embedding) > 0
        print(f"Text embedding dimension: {len(embedding)}")
    
    def test_multimodal_fusion(self, embeddings_service, sample_image):
        """Test multimodal embedding fusion"""
        visual = embeddings_service.create_visual_embedding(sample_image)
        text = embeddings_service.create_text_embedding("Test description")
        
        fused = embeddings_service.create_multimodal_embedding(
            visual, text, fusion_strategy="concat"
        )
        
        assert len(fused) == len(visual) + len(text)
    
    def test_temporal_embeddings(self, embeddings_service, sample_image):
        """Test temporal embedding generation"""
        # Create sequence of embeddings
        embeddings = [
            embeddings_service.create_visual_embedding(sample_image)
            for _ in range(10)
        ]
        timestamps = [i * 0.5 for i in range(10)]
        
        temporal = embeddings_service.create_temporal_embedding(
            embeddings, timestamps, window_size=3
        )
        
        assert len(temporal) == len(embeddings)
        assert all(len(t) > len(embeddings[0]) for t in temporal)  # Should be larger
    
    def test_hierarchical_embeddings(self, embeddings_service, sample_image):
        """Test hierarchical video segmentation"""
        # Simulate 60 seconds of video at 2 fps
        frame_embeddings = [
            embeddings_service.create_visual_embedding(sample_image)
            for _ in range(120)
        ]
        timestamps = [i * 0.5 for i in range(120)]
        
        hierarchy = embeddings_service.create_hierarchical_embeddings(
            frame_embeddings, timestamps, segment_duration=5.0
        )
        
        assert "frames" in hierarchy
        assert "segments" in hierarchy
        assert "scenes" in hierarchy
        assert len(hierarchy["frames"]) == 120
        assert len(hierarchy["segments"]) > 0
        assert len(hierarchy["scenes"]) > 0
        
        print(f"Hierarchy: {len(hierarchy['frames'])} frames, "
              f"{len(hierarchy['segments'])} segments, {len(hierarchy['scenes'])} scenes")
    
    def test_similarity_computation(self, embeddings_service, sample_image):
        """Test similarity computation"""
        embed1 = embeddings_service.create_visual_embedding(sample_image)
        embed2 = embeddings_service.create_visual_embedding(sample_image)  # Same image
        
        # Create slightly different embedding
        embed3 = embed1 + np.random.normal(0, 0.1, embed1.shape)
        
        scores = embeddings_service.compute_similarity(
            embed1, [embed2, embed3], metric="cosine"
        )
        
        # Same image should have higher similarity
        assert scores[0] > scores[1]
        print(f"Similarity scores: {scores}")
    
    def test_temporal_reranking(self, embeddings_service):
        """Test temporal coherence re-ranking"""
        # Simulate search results
        results = [(10, 0.9), (11, 0.7), (50, 0.8), (51, 0.75)]
        timestamps = [i * 0.5 for i in range(100)]
        
        reranked = embeddings_service.rerank_with_temporal_coherence(
            results, timestamps, coherence_weight=0.2
        )
        
        # Results near each other should get boosted
        assert len(reranked) == len(results)
        print(f"Original: {results}")
        print(f"Reranked: {reranked}")


class TestMetrics:
    """Test suite for Metrics service"""
    
    def test_memory_tracking(self, metrics_service):
        """Test memory usage tracking"""
        initial = metrics_service.get_memory_usage()
        
        # Allocate some memory
        large_array = np.random.rand(1000, 1000)
        
        after = metrics_service.get_memory_usage()
        delta = metrics_service.get_peak_memory_delta()
        
        assert after >= initial
        print(f"Memory: {initial:.1f}MB -> {after:.1f}MB (delta: {delta:.1f}MB)")
    
    def test_processing_timer(self, metrics_service):
        """Test processing time tracking"""
        with metrics_service.track_processing_time("Test Operation") as timer:
            time.sleep(0.1)
        
        assert timer.duration >= 0.1
        assert timer.duration < 0.2
    
    def test_answer_quality_evaluation(self, metrics_service):
        """Test answer quality metrics"""
        answer = "The person is walking in the park with a dog"
        ground_truth = "A person walks in the park with their dog"
        
        quality = metrics_service.evaluate_answer_quality(answer, ground_truth)
        
        assert "length" in quality
        assert "word_overlap" in quality
        assert "exact_match" in quality
        assert quality["word_overlap"] > 0.5  # Should have good overlap
        print(f"Quality metrics: {quality}")


class TestIntegration:
    """Integration tests combining multiple services"""
    
    def test_full_video_qa_pipeline(
        self, 
        vision_qa_service, 
        embeddings_service,
        sample_image,
        metrics_service
    ):
        """Test complete VQA pipeline with metrics"""
        with metrics_service.track_processing_time("Full Pipeline") as timer:
            # Generate caption
            caption = vision_qa_service.generate_caption(sample_image)
            
            # Create embeddings
            visual_embed = embeddings_service.create_visual_embedding(sample_image)
            text_embed = embeddings_service.create_text_embedding(caption)
            
            # Fuse modalities
            fused = embeddings_service.create_multimodal_embedding(
                visual_embed, text_embed
            )
            
            # Answer question
            answer, confidence = vision_qa_service.answer_question(
                sample_image, "What is in the image?"
            )
        
        assert timer.duration < 10.0
        assert len(fused) > 0
        assert len(answer) > 0
        
        print(f"Pipeline completed in {timer.duration:.2f}s")
        print(f"Caption: {caption}")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
