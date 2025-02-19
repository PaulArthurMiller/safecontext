"""
Embedding Engine module for generating and managing text embeddings.

This module provides an interface to various embedding models (OpenAI, Sentence-BERT)
with support for batching and caching of embeddings.

Example usage:
    engine = EmbeddingEngine(model_name="text-embedding-3-small")
    texts = ["Hello world", "Another example"]
    embeddings = engine.get_embeddings(texts)

Configuration:
    Required environment variables:
    - OPENAI_API_KEY: For OpenAI models
    
    Or pass api_key directly to the constructor:
    engine = EmbeddingEngine(model_name="text-embedding-3-small", api_key="your-key")
"""

import os
from typing import List, Dict, Optional, Union
import numpy as np
from dataclasses import dataclass
import openai
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
import json
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for the embedding engine."""
    model_name: str
    batch_size: int = 32
    cache_dir: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_retries: int = 3
    timeout: int = 30

class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""
    pass

class ModelAPIError(EmbeddingError):
    """Exception raised for API-related errors."""
    pass

class EmbeddingEngine:
    """
    A class to handle text embedding generation with various model backends.
    
    Supports:
    - OpenAI models (text-embedding-3-small, text-embedding-3-large)
    - Sentence-BERT models (all-MiniLM-L6-v2, etc.)
    
    Features:
    - Automatic batching for efficient processing
    - Result caching
    - Error handling with retries
    - Local file-based embedding persistence
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None
    ):
        """
        Initialize the embedding engine.
        
        Args:
            model_name: Name of the embedding model to use
            api_key: API key for OpenAI (if using OpenAI models)
            config: Optional configuration object
        """
        self.config = config or EmbeddingConfig(model_name=model_name)
        
        # Set up API key for OpenAI if needed
        if "text-embedding" in model_name:
            openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
            self.model = None
            self._get_embeddings = self._get_openai_embeddings
        else:
            # Load local Sentence-BERT model
            try:
                self.model = SentenceTransformer(model_name, device=self.config.device)
                self._get_embeddings = self._get_sbert_embeddings
            except Exception as e:
                raise ModelAPIError(f"Failed to load Sentence-BERT model: {str(e)}")
        
        # Initialize cache if directory is specified
        if self.config.cache_dir:
            os.makedirs(self.config.cache_dir, exist_ok=True)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy.ndarray: Array of embeddings, shape (n_texts, embedding_dim)
            
        Raises:
            ModelAPIError: If there's an error getting embeddings
        """
        try:
            # Check cache first if enabled
            if self.config.cache_dir:
                cached_embeddings = self._load_from_cache(texts)
                if cached_embeddings is not None:
                    return cached_embeddings
            
            # Process in batches
            all_embeddings = []
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                batch_embeddings = self._get_embeddings(batch)
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings)
            
            # Cache results if enabled
            if self.config.cache_dir:
                self._save_to_cache(texts, embeddings)
            
            return embeddings
            
        except Exception as e:
            raise ModelAPIError(f"Error generating embeddings: {str(e)}")
    
    def _get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using OpenAI API."""
        for attempt in range(self.config.max_retries):
            try:
                response = openai.embeddings.create(
                    model=self.config.model_name,
                    input=texts,
                    timeout=self.config.timeout
                )
                return np.array([emb.embedding for emb in response.data])
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise ModelAPIError(f"OpenAI API error after {self.config.max_retries} attempts: {str(e)}")
                logger.warning(f"Retry {attempt + 1}/{self.config.max_retries} after error: {str(e)}")
    
    def _get_sbert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using Sentence-BERT."""
        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            return embeddings
        except Exception as e:
            raise ModelAPIError(f"Sentence-BERT embedding error: {str(e)}")
    
    def _load_from_cache(self, texts: List[str]) -> Optional[np.ndarray]:
        """Load embeddings from cache if available."""
        if not self.config.cache_dir:
            return None
            
        cache_file = Path(self.config.cache_dir) / f"{self.config.model_name}_cache.json"
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            
            # Check if all texts are in cache
            embeddings = []
            for text in texts:
                if text not in cache:
                    return None
                embeddings.append(cache[text])
            
            return np.array(embeddings)
        except Exception as e:
            logger.warning(f"Error loading from cache: {str(e)}")
            return None
    
    def _save_to_cache(self, texts: List[str], embeddings: np.ndarray):
        """Save embeddings to cache."""
        if not self.config.cache_dir:
            return
            
        cache_file = Path(self.config.cache_dir) / f"{self.config.model_name}_cache.json"
        try:
            # Load existing cache
            cache = {}
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            
            # Update cache with new embeddings
            for text, embedding in zip(texts, embeddings):
                cache[text] = embedding.tolist()
            
            # Save updated cache
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")
