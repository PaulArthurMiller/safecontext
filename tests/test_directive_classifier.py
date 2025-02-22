import pytest
import numpy as np
from pathlib import Path
import json
import tempfile
from classification.directive_classifier import (
    DirectiveClassifier,
    ClassifierConfig,
    Label,
    ClassificationResult
)

@pytest.fixture
def embedding_dim():
    return 384  # Common dimension for sentence transformers

@pytest.fixture
def sample_embeddings(embedding_dim):
    return np.random.rand(5, embedding_dim)

@pytest.fixture
def reference_embeddings_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            'embeddings': [[1.0] * 384]  # Single reference embedding
        }, f)
    yield Path(f.name)
    Path(f.name).unlink()  # Cleanup after tests

def test_classifier_initialization(embedding_dim, caplog):
    """Test basic initialization of classifier with fallback behavior"""
    classifier = DirectiveClassifier(embedding_dim=embedding_dim)
    assert classifier.config.model_type == "similarity"
    assert classifier.config.similarity_threshold == 0.6
    assert len(classifier.config.reference_directives) > 0
    
    # Verify warning about fallback embeddings
    assert "No reference_embeddings_path specified" in caplog.text
    assert "Using built-in fallback directive embeddings" in caplog.text
    
    # Verify fallback embeddings were created with correct shape
    expected_size = len(classifier.config.reference_directives)
    assert classifier.reference_embeddings.shape == (expected_size, embedding_dim)

def test_classifier_initialization_no_embedding_dim():
    """Test initialization fails without embedding_dim"""
    with pytest.raises(ValueError, match="embedding_dim must be provided"):
        DirectiveClassifier()

def test_classifier_invalid_model_type(embedding_dim):
    """Test initialization with invalid model type"""
    config = ClassifierConfig(model_type="invalid")
    with pytest.raises(ValueError, match="Unsupported model type"):
        DirectiveClassifier(config=config, embedding_dim=embedding_dim)

def test_ml_classifier_not_implemented(embedding_dim):
    """Test ML classifier raises NotImplementedError"""
    config = ClassifierConfig(model_type="ml", ml_model_path="dummy.pkl")
    with pytest.raises(NotImplementedError):
        DirectiveClassifier(config=config, embedding_dim=embedding_dim)

def test_reference_embeddings_loading(embedding_dim, reference_embeddings_file):
    """Test loading reference embeddings from file"""
    config = ClassifierConfig(
        reference_embeddings_path=str(reference_embeddings_file)
    )
    classifier = DirectiveClassifier(config=config, embedding_dim=embedding_dim)
    assert classifier.reference_embeddings.shape[1] == embedding_dim

def test_invalid_reference_embeddings_path(embedding_dim):
    """Test loading from invalid embeddings path"""
    config = ClassifierConfig(reference_embeddings_path="nonexistent.json")
    with pytest.raises(ValueError, match="Error loading reference embeddings from nonexistent.json"):
        DirectiveClassifier(config=config, embedding_dim=embedding_dim)

def test_custom_reference_directives(embedding_dim):
    """Test initialization with custom reference directives"""
    custom_directives = ["must do", "never allow", "always ensure"]
    config = ClassifierConfig(reference_directives=custom_directives)
    classifier = DirectiveClassifier(config=config, embedding_dim=embedding_dim)
    
    # Verify custom directives were used
    assert classifier.config.reference_directives == custom_directives
    # Verify fallback embeddings match custom directives length
    assert classifier.reference_embeddings.shape == (len(custom_directives), embedding_dim)

def test_empty_reference_directives(embedding_dim):
    """Test initialization with empty reference directives list"""
    config = ClassifierConfig(reference_directives=[])
    classifier = DirectiveClassifier(config=config, embedding_dim=embedding_dim)
    
    # Verify default directives were loaded
    assert len(classifier.config.reference_directives) > 0
    # Verify fallback embeddings match default directives length
    assert classifier.reference_embeddings.shape[0] == len(classifier.config.reference_directives)

def test_classify_single_chunk(embedding_dim):
    """Test classification of a single chunk"""
    classifier = DirectiveClassifier(embedding_dim=embedding_dim)
    embedding = np.ones(embedding_dim)  # Create test embedding
    
    result = classifier.classify_chunk(embedding)
    assert isinstance(result, ClassificationResult)
    assert isinstance(result.label, Label)
    assert isinstance(result.confidence, float)
    assert result.features is not None
    assert "max_similarity" in result.features

def test_classify_multiple_chunks(embedding_dim, sample_embeddings):
    """Test classification of multiple chunks"""
    classifier = DirectiveClassifier(embedding_dim=embedding_dim)
    extra_features = [{"test_feature": i} for i in range(len(sample_embeddings))]
    
    results = classifier.classify_chunks(sample_embeddings, extra_features)
    assert len(results) == len(sample_embeddings)
    for result in results:
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.label, Label)
        assert isinstance(result.confidence, float)
        assert result.features is not None

def test_update_reference_embeddings(embedding_dim, sample_embeddings, tmp_path):
    """Test updating reference embeddings"""
    save_path = tmp_path / "new_embeddings.json"
    config = ClassifierConfig(reference_embeddings_path=str(save_path))
    classifier = DirectiveClassifier(config=config, embedding_dim=embedding_dim)
    
    # Update embeddings
    classifier.update_reference_embeddings(sample_embeddings)
    assert np.array_equal(classifier.reference_embeddings, sample_embeddings)
    
    # Verify file was saved
    assert save_path.exists()
    with save_path.open('r') as f:
        saved_data = json.load(f)
    assert np.array_equal(
        np.array(saved_data['embeddings']),
        sample_embeddings
    )

def test_similarity_classification_threshold(embedding_dim):
    """Test similarity threshold behavior"""
    config = ClassifierConfig(similarity_threshold=0.8)
    classifier = DirectiveClassifier(config=config, embedding_dim=embedding_dim)
    
    # Create an embedding that will have low similarity
    low_sim_embedding = np.zeros(embedding_dim)
    result = classifier.classify_chunk(low_sim_embedding)
    assert result.label == Label.CONTENT
    
    # Create an embedding that will have high similarity
    high_sim_embedding = np.ones(embedding_dim)
    result = classifier.classify_chunk(high_sim_embedding)
    assert result.label == Label.DIRECTIVE
