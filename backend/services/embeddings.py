import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class EmbeddingsService:
    """Vector embeddings and semantic search service using ChromaDB"""
    
    def __init__(
        self, 
        persist_dir: str = "./chroma_db",
        embedding_model: str = "clip-ViT-B-32"
    ):
        """
        Initialize embeddings service
        
        Args:
            persist_dir: Directory to persist ChromaDB
            embedding_model: Sentence transformer model name
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.client = None
        self.collection = None
        
        self._initialize_db()
        self._load_embedding_model()
    
    def _initialize_db(self):
        """Initialize ChromaDB client"""
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="video_frames",
                metadata={"description": "Video frame embeddings"}
            )
            
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _load_embedding_model(self):
        """Load sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def create_text_embedding(self, text: str) -> List[float]:
        """
        Create embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list
        """
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def create_image_embedding(self, image: np.ndarray) -> List[float]:
        """
        Create embedding for image (using CLIP)
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Embedding vector as list
        """
        from PIL import Image
        import cv2
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image)
        embedding = self.embedding_model.encode(pil_image)
        return embedding.tolist()
    
    def add_frame_embedding(
        self,
        video_id: str,
        frame_number: int,
        timestamp: float,
        embedding: List[float],
        metadata: Optional[Dict] = None
    ):
        """
        Add frame embedding to collection
        
        Args:
            video_id: Video ID
            frame_number: Frame number
            timestamp: Timestamp in seconds
            embedding: Embedding vector
            metadata: Additional metadata
        """
        doc_id = f"{video_id}_frame_{frame_number}"
        
        meta = {
            "video_id": video_id,
            "frame_number": frame_number,
            "timestamp": timestamp
        }
        
        if metadata:
            meta.update(metadata)
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[meta]
        )
    
    def add_frame_with_caption(
        self,
        video_id: str,
        frame_number: int,
        timestamp: float,
        image: np.ndarray,
        caption: str
    ):
        """
        Add frame with image and caption embeddings
        
        Args:
            video_id: Video ID
            frame_number: Frame number
            timestamp: Timestamp in seconds
            image: Frame image
            caption: Frame caption/description
        """
        # Create embedding from caption (text is usually better for search)
        embedding = self.create_text_embedding(caption)
        
        metadata = {
            "caption": caption,
            "has_image": True
        }
        
        self.add_frame_embedding(
            video_id, 
            frame_number, 
            timestamp, 
            embedding, 
            metadata
        )
    
    def batch_add_frames(
        self,
        video_id: str,
        frames_data: List[Dict]
    ):
        """
        Batch add multiple frames
        
        Args:
            video_id: Video ID
            frames_data: List of frame data dictionaries
        """
        ids = []
        embeddings = []
        metadatas = []
        
        for frame_data in frames_data:
            frame_num = frame_data["frame_number"]
            timestamp = frame_data["timestamp"]
            caption = frame_data.get("caption", "")
            
            doc_id = f"{video_id}_frame_{frame_num}"
            embedding = self.create_text_embedding(caption)
            
            meta = {
                "video_id": video_id,
                "frame_number": frame_num,
                "timestamp": timestamp,
                "caption": caption
            }
            
            ids.append(doc_id)
            embeddings.append(embedding)
            metadatas.append(meta)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(frames_data)} frames to collection")
    
    def search_frames(
        self,
        query: str,
        video_id: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Search for relevant frames using text query
        
        Args:
            query: Search query
            video_id: Filter by video ID (optional)
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of matching frames with metadata
        """
        # Create query embedding
        query_embedding = self.create_text_embedding(query)
        
        # Build where clause
        where = {"video_id": video_id} if video_id else None
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        
        # Format results
        matches = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                similarity = 1 - distance  # Convert distance to similarity
                
                if similarity >= min_similarity:
                    match = {
                        "id": results["ids"][0][i],
                        "similarity": similarity,
                        "metadata": results["metadatas"][0][i]
                    }
                    matches.append(match)
        
        logger.info(f"Found {len(matches)} matches for query: {query}")
        return matches
    
    def search_by_image(
        self,
        image: np.ndarray,
        video_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar frames using an image
        
        Args:
            image: Query image
            video_id: Filter by video ID (optional)
            top_k: Number of results to return
            
        Returns:
            List of similar frames
        """
        # Create image embedding
        query_embedding = self.create_image_embedding(image)
        
        # Build where clause
        where = {"video_id": video_id} if video_id else None
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        
        # Format results
        matches = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                similarity = 1 - distance
                
                match = {
                    "id": results["ids"][0][i],
                    "similarity": similarity,
                    "metadata": results["metadatas"][0][i]
                }
                matches.append(match)
        
        return matches
    
    def get_frame_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Get frame by document ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Frame metadata or None
        """
        results = self.collection.get(ids=[doc_id])
        
        if results["ids"]:
            return {
                "id": results["ids"][0],
                "metadata": results["metadatas"][0]
            }
        return None
    
    def delete_video_frames(self, video_id: str):
        """
        Delete all frames for a video
        
        Args:
            video_id: Video ID
        """
        self.collection.delete(where={"video_id": video_id})
        logger.info(f"Deleted frames for video: {video_id}")
    
    def get_video_frame_count(self, video_id: str) -> int:
        """
        Get number of frames stored for a video
        
        Args:
            video_id: Video ID
            
        Returns:
            Frame count
        """
        results = self.collection.get(where={"video_id": video_id})
        return len(results["ids"]) if results["ids"] else 0
    
    def find_temporal_neighbors(
        self,
        video_id: str,
        timestamp: float,
        window: float = 5.0
    ) -> List[Dict]:
        """
        Find frames near a specific timestamp
        
        Args:
            video_id: Video ID
            timestamp: Target timestamp
            window: Time window in seconds
            
        Returns:
            List of nearby frames
        """
        # Get all frames for video
        results = self.collection.get(where={"video_id": video_id})
        
        if not results["ids"]:
            return []
        
        # Filter by time window
        nearby_frames = []
        for i, metadata in enumerate(results["metadatas"]):
            frame_time = metadata["timestamp"]
            if abs(frame_time - timestamp) <= window:
                nearby_frames.append({
                    "id": results["ids"][i],
                    "metadata": metadata,
                    "time_diff": abs(frame_time - timestamp)
                })
        
        # Sort by time difference
        nearby_frames.sort(key=lambda x: x["time_diff"])
        
        return nearby_frames
    
    def cluster_similar_frames(
        self,
        video_id: str,
        n_clusters: int = 10
    ) -> Dict[int, List[Dict]]:
        """
        Cluster frames by visual similarity
        
        Args:
            video_id: Video ID
            n_clusters: Number of clusters
            
        Returns:
            Dictionary mapping cluster ID to frames
        """
        from sklearn.cluster import KMeans
        
        # Get all frames for video
        results = self.collection.get(
            where={"video_id": video_id},
            include=["embeddings", "metadatas"]
        )
        
        if not results["ids"] or len(results["ids"]) < n_clusters:
            return {}
        
        # Cluster embeddings
        embeddings = np.array(results["embeddings"])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Group by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            
            clusters[label].append({
                "id": results["ids"][i],
                "metadata": results["metadatas"][i]
            })
        
        logger.info(f"Clustered {len(results['ids'])} frames into {n_clusters} clusters")
        return clusters
