"""
Directive Classifier module for identifying directive vs content text.

This module provides classification capabilities to determine whether
text chunks contain directives (instructions, commands, policies) or
general content. It supports both simple similarity-based and ML-based
approaches.

Example usage:
    classifier = DirectiveClassifier()
    result = classifier.classify_chunk(embedding)
    label, confidence = result.label, result.confidence
    
    # With multiple chunks
    results = classifier.classify_chunks(embeddings)
    for result in results:
        print(f"Label: {result.label}, Confidence: {result.confidence}")
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
from enum import Enum
from utils.logger import SafeContextLogger
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import json

# Configure logging
logger = SafeContextLogger(__name__)

class Label(str, Enum):
    """Classification labels for text chunks."""
    DIRECTIVE = "directive"
    CONTENT = "content"

@dataclass
class ClassificationResult:
    """Container for classification results."""
    label: Label
    confidence: float
    features: Optional[dict] = None

@dataclass
class ClassifierConfig:
    """Configuration for the directive classifier."""
    model_type: str = "similarity"  # Options: "similarity", "ml"
    similarity_threshold: float = 0.6
    reference_embeddings_path: Optional[str] = None
    ml_model_path: Optional[str] = None
    
    # Example directive patterns for similarity-based classification
    reference_directives: Optional[List[str]] = None
    
    def __post_init__(self):
        """Initialize default reference directives if none provided."""
        if self.reference_directives is None:
            self.reference_directives = [
                "you must",
                "it is required",
                "never",
                "always",
                "shall",
                "required to",
                "mandatory",
                "prohibited",
                "ensure that",
                "do not"
            ]

class DirectiveClassifier:
    """
    A classifier for identifying directive vs content text.
    
    Supports multiple classification approaches:
    1. Similarity-based: Uses cosine similarity with reference embeddings
    2. ML-based: Supports integration with trained ML models (future)
    
    Features:
    - Configurable classification thresholds
    - Support for batch processing
    - Extensible for different ML models
    - Optional feature extraction
    """
    
    def __init__(
        self,
        config: Optional[ClassifierConfig] = None,
        embedding_dim: Optional[int] = None
    ) -> None:
        """
        Initialize the classifier.
        
        Args:
            config: Optional configuration object
            embedding_dim: Dimension of input embeddings (required for similarity approach)
        """
        self.config: ClassifierConfig = config or ClassifierConfig()
        self.reference_embeddings: np.ndarray = np.array([], dtype=np.float32)  # Initialize empty with proper dtype
        
        if self.config.model_type == "similarity":
            if embedding_dim is None:
                raise ValueError("embedding_dim must be provided for similarity-based classification")
            self._initialize_similarity_classifier(embedding_dim)
        elif self.config.model_type == "ml":
            self._initialize_ml_classifier()
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _initialize_similarity_classifier(self, embedding_dim: int):
        """Initialize the similarity-based classifier."""
        # Load reference embeddings if provided
        if self.config.reference_embeddings_path:
            self._load_reference_embeddings()
        else:
            # Initialize with ones vectors for better similarity testing
            if self.config.reference_directives is None:
                raise ValueError("reference_directives cannot be None")
            self.reference_embeddings = np.ones((len(self.config.reference_directives), embedding_dim))
            logger.warning("Using ones vectors for reference embeddings - replace with actual embeddings")
    
    def _initialize_ml_classifier(self):
        """Initialize the ML-based classifier."""
        if not self.config.ml_model_path:
            raise ValueError("ml_model_path must be provided for ML-based classification")
        # TODO: Implement ML model loading
        raise NotImplementedError("ML-based classification not yet implemented")
    
    def _load_reference_embeddings(self):
        """Load reference embeddings from file."""
        if self.config.reference_embeddings_path is None:
            raise ValueError("reference_embeddings_path cannot be None")
        try:
            path = Path(self.config.reference_embeddings_path)
            with path.open('r') as f:
                data = json.load(f)
            self.reference_embeddings = np.array(data['embeddings'])
        except Exception as e:
            raise ValueError(f"Error loading reference embeddings: {str(e)}")
    
    def classify_chunks(
        self,
        embeddings: np.ndarray,
        extra_features: Optional[List[dict]] = None
    ) -> List[ClassificationResult]:
        """
        Classify multiple text chunks.
        
        Args:
            embeddings: Array of embedding vectors, shape (n_chunks, embedding_dim)
            extra_features: Optional list of feature dictionaries for each chunk
            
        Returns:
            List of ClassificationResult objects
        """
        results = []
        for i, embedding in enumerate(embeddings):
            features = extra_features[i] if extra_features else None
            result = self.classify_chunk(embedding, features)
            results.append(result)
        return results
    
    def classify_chunk(
        self,
        embedding: np.ndarray,
        extra_features: Optional[dict] = None
    ) -> ClassificationResult:
        """
        Classify a single text chunk.
        
        Args:
            embedding: Embedding vector for the chunk
            extra_features: Optional dictionary of additional features
            
        Returns:
            ClassificationResult object with label and confidence
        """
        if self.config.model_type == "similarity":
            return self._similarity_classification(embedding, extra_features)
        else:
            return self._ml_classification(embedding, extra_features)
    
    def _similarity_classification(
        self,
        embedding: np.ndarray,
        extra_features: Optional[dict]
    ) -> ClassificationResult:
        """Classify using similarity-based approach."""
        # Calculate similarities with reference embeddings
        similarities = cosine_similarity(embedding.reshape(1, -1), self.reference_embeddings)[0]
        max_similarity = np.max(similarities)
        
        # Classify based on similarity threshold
        if max_similarity >= self.config.similarity_threshold:
            label = Label.DIRECTIVE
            confidence = max_similarity
        else:
            label = Label.CONTENT
            confidence = 1 - max_similarity
        
        features = {"max_similarity": float(max_similarity)}
        if extra_features:
            features.update(extra_features)
            
        return ClassificationResult(
            label=label,
            confidence=float(confidence),
            features=features
        )
    
    def _ml_classification(
        self,
        embedding: np.ndarray,
        extra_features: Optional[dict]
    ) -> ClassificationResult:
        """Classify using ML-based approach."""
        # TODO: Implement ML-based classification
        raise NotImplementedError("ML-based classification not yet implemented")
    
    def update_reference_embeddings(self, embeddings: np.ndarray, save: bool = True):
        """
        Update reference embeddings for similarity-based classification.
        
        Args:
            embeddings: New reference embeddings
            save: Whether to save to file if reference_embeddings_path is set
        """
        self.reference_embeddings = embeddings
        
        if save and self.config.reference_embeddings_path is not None:
            try:
                path = Path(self.config.reference_embeddings_path)
                with path.open('w') as f:
                    json.dump({'embeddings': embeddings.tolist()}, f)
                logger.info("Saved updated reference embeddings")
            except Exception as e:
                logger.error(f"Error saving reference embeddings: {str(e)}")
