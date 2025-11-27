"""
Model comparison and A/B testing service.
Allows comparing different models and selecting the best for each task.
"""
import time
import logging
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Supported model types"""
    VQA = "vqa"
    CAPTION = "caption"
    DETECTION = "detection"
    EMBEDDING = "embedding"


class ModelRegistry:
    """Registry for multiple model implementations"""
    
    def __init__(self):
        self.models = {
            ModelType.VQA: {},
            ModelType.CAPTION: {},
            ModelType.DETECTION: {},
            ModelType.EMBEDDING: {}
        }
        self.default_models = {}
    
    def register_model(
        self,
        model_type: ModelType,
        model_name: str,
        model_fn: Callable,
        is_default: bool = False
    ):
        """Register a model implementation"""
        self.models[model_type][model_name] = model_fn
        
        if is_default or not self.default_models.get(model_type):
            self.default_models[model_type] = model_name
        
        logger.info(f"Registered {model_type} model: {model_name}")
    
    def get_model(self, model_type: ModelType, model_name: Optional[str] = None):
        """Get a model by type and name"""
        if model_name is None:
            model_name = self.default_models.get(model_type)
        
        if model_name not in self.models[model_type]:
            raise ValueError(f"Model {model_name} not found for type {model_type}")
        
        return self.models[model_type][model_name]
    
    def list_models(self, model_type: ModelType) -> List[str]:
        """List all registered models of a type"""
        return list(self.models[model_type].keys())


class ModelComparator:
    """Compare performance of different models"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.comparison_history = []
    
    def compare_models(
        self,
        model_type: ModelType,
        input_data: any,
        model_names: Optional[List[str]] = None,
        ground_truth: Optional[any] = None
    ) -> Dict:
        """
        Compare multiple models on the same input.
        
        Args:
            model_type: Type of models to compare
            input_data: Input for the models
            model_names: List of model names to compare (None = all)
            ground_truth: Optional ground truth for accuracy
            
        Returns:
            Comparison results
        """
        if model_names is None:
            model_names = self.registry.list_models(model_type)
        
        results = {}
        
        for name in model_names:
            try:
                model_fn = self.registry.get_model(model_type, name)
                
                # Time the inference
                start_time = time.time()
                output = model_fn(input_data)
                inference_time = time.time() - start_time
                
                # Calculate accuracy if ground truth provided
                accuracy = None
                if ground_truth is not None:
                    accuracy = self._calculate_accuracy(output, ground_truth)
                
                results[name] = {
                    "output": output,
                    "inference_time": inference_time,
                    "accuracy": accuracy
                }
                
                logger.info(f"{name}: {inference_time:.3f}s" + 
                          (f", acc: {accuracy:.2f}" if accuracy else ""))
                
            except Exception as e:
                logger.error(f"Error comparing model {name}: {e}")
                results[name] = {"error": str(e)}
        
        # Record comparison
        comparison = {
            "model_type": model_type,
            "timestamp": time.time(),
            "results": results
        }
        self.comparison_history.append(comparison)
        
        return results
    
    def _calculate_accuracy(self, output: any, ground_truth: any) -> float:
        """Calculate accuracy score"""
        if isinstance(output, str) and isinstance(ground_truth, str):
            # Text similarity (simple word overlap)
            output_words = set(output.lower().split())
            truth_words = set(ground_truth.lower().split())
            
            if not truth_words:
                return 0.0
            
            overlap = len(output_words & truth_words) / len(truth_words)
            return overlap
        
        elif isinstance(output, (list, np.ndarray)) and isinstance(ground_truth, (list, np.ndarray)):
            # Numerical similarity
            output_arr = np.array(output)
            truth_arr = np.array(ground_truth)
            
            # Cosine similarity
            dot = np.dot(output_arr, truth_arr)
            norm = np.linalg.norm(output_arr) * np.linalg.norm(truth_arr)
            return float(dot / (norm + 1e-8))
        
        return 0.0
    
    def get_best_model(
        self,
        model_type: ModelType,
        metric: str = "inference_time",
        minimize: bool = True
    ) -> str:
        """
        Get the best model based on historical comparisons.
        
        Args:
            model_type: Type of model
            metric: Metric to optimize ('inference_time', 'accuracy')
            minimize: Whether to minimize the metric
            
        Returns:
            Name of best model
        """
        # Filter comparisons for this model type
        relevant = [c for c in self.comparison_history if c["model_type"] == model_type]
        
        if not relevant:
            return self.registry.default_models.get(model_type)
        
        # Aggregate scores by model
        model_scores = {}
        
        for comparison in relevant:
            for model_name, result in comparison["results"].items():
                if "error" in result:
                    continue
                
                if model_name not in model_scores:
                    model_scores[model_name] = []
                
                score = result.get(metric)
                if score is not None:
                    model_scores[model_name].append(score)
        
        if not model_scores:
            return self.registry.default_models.get(model_type)
        
        # Calculate average scores
        avg_scores = {
            name: sum(scores) / len(scores)
            for name, scores in model_scores.items()
        }
        
        # Find best
        if minimize:
            best = min(avg_scores.items(), key=lambda x: x[1])
        else:
            best = max(avg_scores.items(), key=lambda x: x[1])
        
        logger.info(f"Best model for {model_type} by {metric}: {best[0]} ({best[1]:.3f})")
        
        return best[0]
    
    def get_statistics(self) -> Dict:
        """Get comparison statistics"""
        stats = {
            "total_comparisons": len(self.comparison_history),
            "by_type": {}
        }
        
        for model_type in ModelType:
            type_comparisons = [c for c in self.comparison_history if c["model_type"] == model_type]
            
            if type_comparisons:
                stats["by_type"][model_type.value] = {
                    "count": len(type_comparisons),
                    "models_tested": len(set(
                        model_name 
                        for c in type_comparisons 
                        for model_name in c["results"].keys()
                    ))
                }
        
        return stats


# Global instances
model_registry = ModelRegistry()
model_comparator = ModelComparator(model_registry)


def register_default_models(
    vision_qa,
    object_detector,
    embeddings_service
):
    """Register default model implementations"""
    
    # VQA models
    model_registry.register_model(
        ModelType.VQA,
        "blip-vqa-base",
        lambda input_data: vision_qa.answer_question(
            input_data["image"], 
            input_data["question"]
        ),
        is_default=True
    )
    
    # Caption models
    model_registry.register_model(
        ModelType.CAPTION,
        "blip-caption-base",
        lambda image: vision_qa.generate_caption(image),
        is_default=True
    )
    
    # Detection models
    model_registry.register_model(
        ModelType.DETECTION,
        "yolov8n",
        lambda image: object_detector.detect_objects(image),
        is_default=True
    )
    
    # Embedding models
    model_registry.register_model(
        ModelType.EMBEDDING,
        "clip-vit-b-32",
        lambda image: embeddings_service.create_visual_embedding(image),
        is_default=True
    )
    
    logger.info("✓ Default models registered")
