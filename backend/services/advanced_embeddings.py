"""
Advanced embeddings service with temporal awareness and multi-modal fusion.
Implements hierarchical video segmentation and sophisticated retrieval strategies.
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from PIL import Image
import cv2
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AdvancedEmbeddingsService:
    """
    Advanced embeddings with temporal and hierarchical features.
    Combines visual, textual, and temporal information for better video understanding.
    """
    
    def __init__(
        self,
        visual_model: str = "clip-ViT-B-32",
        text_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize advanced embeddings service
        
        Args:
            visual_model: Model for visual embeddings
            text_model: Model for text embeddings
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self.visual_encoder = SentenceTransformer(visual_model, device=self.device)
        self.text_encoder = SentenceTransformer(text_model, device=self.device)
        
        logger.info("✓ Advanced embeddings service initialized")
    
    def create_visual_embedding(self, image: np.ndarray) -> np.ndarray:
        """Create visual embedding from image"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image)
        embedding = self.visual_encoder.encode(pil_image, convert_to_numpy=True)
        return embedding
    
    def create_text_embedding(self, text: str) -> np.ndarray:
        """Create text embedding"""
        embedding = self.text_encoder.encode(text, convert_to_numpy=True)
        return embedding
    
    def create_temporal_embedding(
        self,
        embeddings: List[np.ndarray],
        timestamps: List[float],
        window_size: int = 5
    ) -> List[np.ndarray]:
        """
        Create temporal-aware embeddings using sliding window.
        Captures motion and change over time.
        
        Args:
            embeddings: List of frame embeddings
            timestamps: Corresponding timestamps
            window_size: Size of temporal window
            
        Returns:
            Temporal-aware embeddings
        """
        temporal_embeds = []
        
        for i in range(len(embeddings)):
            # Get window around current frame
            start = max(0, i - window_size // 2)
            end = min(len(embeddings), i + window_size // 2 + 1)
            
            window = embeddings[start:end]
            
            # Compute temporal features
            if len(window) > 1:
                # Average embedding in window
                avg_embed = np.mean(window, axis=0)
                
                # Compute change from previous frame
                if i > 0:
                    delta = embeddings[i] - embeddings[i-1]
                else:
                    delta = np.zeros_like(embeddings[i])
                
                # Concatenate: [current, average, delta]
                temporal_embed = np.concatenate([
                    embeddings[i],
                    avg_embed,
                    delta
                ])
            else:
                # First frame: just use current embedding
                temporal_embed = np.concatenate([
                    embeddings[i],
                    embeddings[i],
                    np.zeros_like(embeddings[i])
                ])
            
            temporal_embeds.append(temporal_embed)
        
        return temporal_embeds
    
    def create_multimodal_embedding(
        self,
        visual_embed: np.ndarray,
        text_embed: np.ndarray,
        audio_embed: Optional[np.ndarray] = None,
        fusion_strategy: str = "concat"
    ) -> np.ndarray:
        """
        Fuse multiple modalities into single embedding.
        
        Args:
            visual_embed: Visual embedding
            text_embed: Text embedding
            audio_embed: Optional audio embedding
            fusion_strategy: 'concat', 'weighted_sum', or 'attention'
            
        Returns:
            Fused multimodal embedding
        """
        if fusion_strategy == "concat":
            # Simple concatenation
            embeds = [visual_embed, text_embed]
            if audio_embed is not None:
                embeds.append(audio_embed)
            return np.concatenate(embeds)
        
        elif fusion_strategy == "weighted_sum":
            # Normalize to same dimension and weighted average
            visual_norm = visual_embed / (np.linalg.norm(visual_embed) + 1e-8)
            text_norm = text_embed / (np.linalg.norm(text_embed) + 1e-8)
            
            # Visual gets more weight for video understanding
            weights = [0.6, 0.4] if audio_embed is None else [0.5, 0.3, 0.2]
            
            # Pad to same dimension
            max_dim = max(len(visual_embed), len(text_embed))
            visual_padded = np.pad(visual_norm, (0, max_dim - len(visual_norm)))
            text_padded = np.pad(text_norm, (0, max_dim - len(text_norm)))
            
            fused = weights[0] * visual_padded + weights[1] * text_padded
            
            if audio_embed is not None:
                audio_norm = audio_embed / (np.linalg.norm(audio_embed) + 1e-8)
                audio_padded = np.pad(audio_norm, (0, max_dim - len(audio_norm)))
                fused += weights[2] * audio_padded
            
            return fused
        
        else:
            # Default to concat
            return np.concatenate([visual_embed, text_embed])
    
    def create_hierarchical_embeddings(
        self,
        frame_embeddings: List[np.ndarray],
        timestamps: List[float],
        segment_duration: float = 5.0
    ) -> Dict[str, List]:
        """
        Create hierarchical embeddings at multiple temporal scales.
        - Frame level: Individual frames
        - Segment level: Short clips (5s)
        - Scene level: Longer sequences (30s)
        
        Args:
            frame_embeddings: List of frame embeddings
            timestamps: Corresponding timestamps
            segment_duration: Duration of segments in seconds
            
        Returns:
            Dictionary with hierarchical embeddings
        """
        hierarchy = {
            "frames": frame_embeddings,
            "segments": [],
            "scenes": []
        }
        
        if not frame_embeddings:
            return hierarchy
        
        # Create segment-level embeddings (5s clips)
        current_segment = []
        segment_start_time = timestamps[0]
        
        for i, (embed, ts) in enumerate(zip(frame_embeddings, timestamps)):
            current_segment.append(embed)
            
            # Check if segment duration reached
            if ts - segment_start_time >= segment_duration or i == len(timestamps) - 1:
                # Average frames in segment
                segment_embed = np.mean(current_segment, axis=0)
                hierarchy["segments"].append({
                    "embedding": segment_embed,
                    "start_time": segment_start_time,
                    "end_time": ts,
                    "frame_count": len(current_segment)
                })
                
                # Reset for next segment
                current_segment = []
                if i < len(timestamps) - 1:
                    segment_start_time = timestamps[i + 1]
        
        # Create scene-level embeddings (30s clips)
        scene_duration = 30.0
        current_scene = []
        scene_start_time = timestamps[0]
        
        for i, (embed, ts) in enumerate(zip(frame_embeddings, timestamps)):
            current_scene.append(embed)
            
            if ts - scene_start_time >= scene_duration or i == len(timestamps) - 1:
                scene_embed = np.mean(current_scene, axis=0)
                hierarchy["scenes"].append({
                    "embedding": scene_embed,
                    "start_time": scene_start_time,
                    "end_time": ts,
                    "frame_count": len(current_scene)
                })
                
                current_scene = []
                if i < len(timestamps) - 1:
                    scene_start_time = timestamps[i + 1]
        
        logger.info(f"Created hierarchical embeddings: {len(hierarchy['frames'])} frames, "
                   f"{len(hierarchy['segments'])} segments, {len(hierarchy['scenes'])} scenes")
        
        return hierarchy
    
    def compute_similarity(
        self,
        query_embed: np.ndarray,
        candidate_embeds: List[np.ndarray],
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute similarity between query and candidates.
        
        Args:
            query_embed: Query embedding
            candidate_embeds: List of candidate embeddings
            metric: 'cosine', 'euclidean', or 'dot'
            
        Returns:
            Similarity scores
        """
        candidates = np.array(candidate_embeds)
        
        if metric == "cosine":
            # Cosine similarity
            query_norm = query_embed / (np.linalg.norm(query_embed) + 1e-8)
            candidates_norm = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)
            scores = np.dot(candidates_norm, query_norm)
        
        elif metric == "euclidean":
            # Negative euclidean distance (higher = more similar)
            distances = np.linalg.norm(candidates - query_embed, axis=1)
            scores = -distances
        
        elif metric == "dot":
            # Dot product
            scores = np.dot(candidates, query_embed)
        
        else:
            scores = np.dot(candidates, query_embed)
        
        return scores
    
    def rerank_with_temporal_coherence(
        self,
        results: List[Tuple[int, float]],
        timestamps: List[float],
        coherence_weight: float = 0.2
    ) -> List[Tuple[int, float]]:
        """
        Re-rank search results considering temporal coherence.
        Boost results that are temporally close to other high-scoring results.
        
        Args:
            results: List of (frame_idx, score) tuples
            timestamps: Frame timestamps
            coherence_weight: Weight for temporal coherence boost
            
        Returns:
            Re-ranked results
        """
        if len(results) <= 1:
            return results
        
        # Sort by score
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        
        # Compute temporal coherence boost
        boosted_results = []
        for idx, score in sorted_results:
            boost = 0.0
            ts = timestamps[idx]
            
            # Check proximity to other high-scoring frames
            for other_idx, other_score in sorted_results[:10]:  # Top 10
                if other_idx != idx:
                    other_ts = timestamps[other_idx]
                    time_diff = abs(ts - other_ts)
                    
                    # Boost if within 5 seconds
                    if time_diff < 5.0:
                        proximity_factor = 1.0 - (time_diff / 5.0)
                        boost += other_score * proximity_factor
            
            # Apply boost
            new_score = score + coherence_weight * boost
            boosted_results.append((idx, new_score))
        
        # Re-sort with boosted scores
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        
        return boosted_results
