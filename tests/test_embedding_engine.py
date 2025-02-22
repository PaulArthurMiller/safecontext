import pytest
import numpy as np
from pathlib import Path
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from embeddings.embedding_engine import (
    EmbeddingEngine,
    EmbeddingConfig,
    EmbeddingError,
    ModelAPIError
)

# Test fixtures
@pytest.fixture
def temp_cache_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def mock_openai():
    with patch('embeddings.embedding_engine.openai') as mock:
        # Mock successful response
        mock_data = [
            MagicMock(embedding=[0.1, 0.2, 0.3])
        ]
        mock_response = MagicMock(data=mock_data)
        mock.embeddings.create.return_value = mock_response
        yield mock

@pytest.fixture
def mock_sentence_transformer():
    with patch('embeddings.embedding_engine.SentenceTransformer') as mock:
        instance = MagicMock()
        instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock.return_value = instance
        yield mock

# Test initialization
def test_init_openai():
    """Test initialization with OpenAI model"""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        engine = EmbeddingEngine(model_name="text-embedding-3-small")
        assert engine.config.model_name == "text-embedding-3-small"
        assert engine.model is None  # OpenAI doesn't use local model

def test_init_sbert():
    """Test initialization with Sentence-BERT model"""
    with patch('embeddings.embedding_engine.SentenceTransformer'):
        engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
        assert engine.config.model_name == "all-MiniLM-L6-v2"
        assert engine.model is not None

def test_init_no_api_key():
    """Test initialization fails without API key for OpenAI"""
    with patch.dict(os.environ, {'OPENAI_API_KEY': ''}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API key must be provided"):
            EmbeddingEngine(model_name="text-embedding-3-small")

def test_init_invalid_sbert():
    """Test initialization fails with invalid SBERT model"""
    with patch('embeddings.embedding_engine.SentenceTransformer', side_effect=Exception("Model not found")):
        with pytest.raises(ModelAPIError, match="Failed to load Sentence-BERT model"):
            EmbeddingEngine(model_name="invalid-model")

# Test embedding generation
def test_get_embeddings_openai(mock_openai):
    """Test getting embeddings via OpenAI"""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        engine = EmbeddingEngine(model_name="text-embedding-3-small")
        texts = ["Hello world"]
        embeddings = engine.get_embeddings(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 3)  # Based on mock response
        mock_openai.embeddings.create.assert_called_once()

def test_get_embeddings_sbert(mock_sentence_transformer):
    """Test getting embeddings via Sentence-BERT"""
    engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    texts = ["Hello world"]
    embeddings = engine.get_embeddings(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 3)  # Based on mock response
    engine.model.encode.assert_called_once()

def test_get_embeddings_empty_input():
    """Test handling of empty input"""
    engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    with pytest.raises(ValueError, match="Input texts list cannot be empty"):
        engine.get_embeddings([])

def test_get_embeddings_invalid_input():
    """Test handling of invalid input"""
    engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    with pytest.raises(ValueError, match="All inputs must be non-empty strings"):
        engine.get_embeddings(["", "test"])

# Test caching
def test_cache_creation(temp_cache_dir):
    """Test cache directory creation"""
    config = EmbeddingConfig(model_name="test-model", cache_dir=temp_cache_dir)
    EmbeddingEngine(config=config, model_name="all-MiniLM-L6-v2")
    assert Path(temp_cache_dir).exists()

def test_cache_save_load(temp_cache_dir, mock_sentence_transformer):
    """Test saving and loading from cache"""
    config = EmbeddingConfig(
        model_name="test-model",
        cache_dir=temp_cache_dir
    )
    engine = EmbeddingEngine(config=config, model_name="all-MiniLM-L6-v2")
    
    # First request should use model
    texts = ["Hello world"]
    first_embeddings = engine.get_embeddings(texts)
    
    # Second request should use cache
    engine.model.encode.reset_mock()
    second_embeddings = engine.get_embeddings(texts)
    
    assert np.array_equal(first_embeddings, second_embeddings)
    engine.model.encode.assert_not_called()  # Should use cache

def test_cache_size_limit(temp_cache_dir, mock_sentence_transformer):
    """Test cache size limiting"""
    config = EmbeddingConfig(
        model_name="test-model",
        cache_dir=temp_cache_dir,
        max_cache_size=2
    )
    engine = EmbeddingEngine(config=config, model_name="all-MiniLM-L6-v2")
    
    # Add three items to trigger size limit
    texts = [f"Text {i}" for i in range(3)]
    engine.get_embeddings(texts)
    
    # Check cache file
    cache_file = Path(temp_cache_dir) / "test-model_cache.json"
    with open(cache_file, 'r') as f:
        cache = json.load(f)
    assert len(cache) == 2  # Should keep only latest 2 entries

def test_clear_cache(temp_cache_dir, mock_sentence_transformer):
    """Test cache clearing"""
    config = EmbeddingConfig(
        model_name="test-model",
        cache_dir=temp_cache_dir
    )
    engine = EmbeddingEngine(config=config, model_name="all-MiniLM-L6-v2")
    
    # Add some items to cache
    texts = ["Hello world"]
    engine.get_embeddings(texts)
    
    # Clear cache
    engine.clear_cache()
    cache_file = Path(temp_cache_dir) / "test-model_cache.json"
    assert not cache_file.exists()

# Test error handling
def test_openai_retry(mock_openai):
    """Test OpenAI API retry behavior"""
    mock_openai.embeddings.create.side_effect = [
        Exception("API Error"),  # First attempt fails
        Exception("API Error"),  # Second attempt fails
        MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])])  # Third attempt succeeds
    ]
    
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        engine = EmbeddingEngine(model_name="text-embedding-3-small")
        embeddings = engine.get_embeddings(["Hello world"])
        assert isinstance(embeddings, np.ndarray)
        assert mock_openai.embeddings.create.call_count == 3

def test_openai_max_retries(mock_openai):
    """Test OpenAI API max retries exceeded"""
    mock_openai.embeddings.create.side_effect = Exception("API Error")
    
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        engine = EmbeddingEngine(model_name="text-embedding-3-small")
        with pytest.raises(ModelAPIError, match="OpenAI API error after .* attempts"):
            engine.get_embeddings(["Hello world"])

def test_sbert_error(mock_sentence_transformer):
    """Test Sentence-BERT error handling"""
    engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    engine.model.encode.side_effect = Exception("SBERT Error")
    
    with pytest.raises(ModelAPIError, match="Sentence-BERT embedding error"):
        engine.get_embeddings(["Hello world"])

# Test cleanup
def test_cleanup():
    """Test cleanup on deletion"""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        engine = EmbeddingEngine(model_name="text-embedding-3-small")
        del engine
        assert openai.api_key is None  # API key should be cleared
