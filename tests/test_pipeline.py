import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np

from pipeline import SafeContextPipeline
from classification.directive_classifier import Label, ClassificationResult

@pytest.fixture
def mock_pipeline_components():
    """
    Fixture that patches the DocumentParser, TextChunker, EmbeddingEngine,
    DirectiveClassifier, and ContextStripper used by SafeContextPipeline.
    Yields a tuple of (parser_mock, chunker_mock, embedding_mock, classifier_mock, stripper_mock).
    """
    with patch("pipeline.DocumentParser", autospec=True) as parser_cls, \
         patch("pipeline.TextChunker", autospec=True) as chunker_cls, \
         patch("pipeline.EmbeddingEngine", autospec=True) as embedding_cls, \
         patch("pipeline.DirectiveClassifier", autospec=True) as classifier_cls, \
         patch("pipeline.ContextStripper", autospec=True) as stripper_cls:

        # Create instances of the mocks
        parser_mock = parser_cls.return_value
        chunker_mock = chunker_cls.return_value
        embedding_mock = embedding_cls.return_value
        classifier_mock = classifier_cls.return_value
        stripper_mock = stripper_cls.return_value

        yield (parser_mock, chunker_mock, embedding_mock, classifier_mock, stripper_mock)

def test_pipeline_end_to_end(mock_pipeline_components):
    """
    Test that SafeContextPipeline processes input by calling parse -> chunk -> embed -> classify -> sanitize,
    and returns the expected final text.
    """
    (parser_mock, chunker_mock, embedding_mock, classifier_mock, stripper_mock) = mock_pipeline_components

    # 1. Setup mock return values for each stage.

    # 1. Make sure pipeline picks up the input as a Path
    input_path_str = "./fake_file.txt"  # has a slash
    # 2. Patch path.exists to return True so parser tries to parse
    with patch.object(Path, "exists", return_value=True):
        parser_mock.parse.return_value = "This is some raw text"

    # Stage 2: chunker returns a list of chunk-like objects
    # We can simulate chunk objects with simple MagicMock or named tuples
    chunk1 = MagicMock(text="Chunk 1 text")
    chunk2 = MagicMock(text="Chunk 2 text")
    chunker_mock.chunk_text.return_value = [chunk1, chunk2]

    # Stage 3: embedding engine returns an np.array for each chunk
    embeddings_array = np.array([[0.1, 0.2], [0.3, 0.4]])
    embedding_mock.get_embeddings.return_value = embeddings_array

    # Stage 4: classifier returns a ClassificationResult list, one per chunk
    # Let’s say chunk1 is directive, chunk2 is content
    classification_results = [
        ClassificationResult(label=Label.DIRECTIVE, confidence=0.9, features=None),
        ClassificationResult(label=Label.CONTENT, confidence=0.1, features=None),
    ]
    classifier_mock.classify_chunks.return_value = classification_results

    # Stage 5: context stripper returns sanitized strings for each chunk
    # Suppose it strips out directives from chunk 1 and leaves chunk 2
    stripper_mock.sanitize.side_effect = ["Sanitized chunk 1", "Sanitized chunk 2"]

    # 2. Create the pipeline and process input
    pipeline = SafeContextPipeline(config_path=None)
    result = pipeline.process_input(input_path_str)
    
    # Now you can check parse was called with Path("./fake_file.txt")
    parser_mock.parse.assert_called_once_with(Path("./fake_file.txt"))

    # Chunker was called with the raw text from parser
    chunker_mock.chunk_text.assert_called_once_with("This is some raw text")

    # Embedding engine was called with the chunk texts
    embedding_mock.get_embeddings.assert_called_once_with(["Chunk 1 text", "Chunk 2 text"])

    # Classifier was called with the embeddings
    classifier_mock.classify_chunks.assert_called_once_with(embeddings_array)

    # Stripper was called once per chunk
    assert stripper_mock.sanitize.call_count == 2
    # The first call had directive_score=0.9, the second had directive_score=0.0 (for content)
    (first_chunk_text, first_score) = stripper_mock.sanitize.call_args_list[0][0]
    assert first_chunk_text == "Chunk 1 text"
    assert first_score == 0.9

    (second_chunk_text, second_score) = stripper_mock.sanitize.call_args_list[1][0]
    assert second_chunk_text == "Chunk 2 text"
    assert second_score == 0.0  # Because that chunk is labeled CONTENT

    # 4. Final output should be joined sanitized text
    assert result == "Sanitized chunk 1 Sanitized chunk 2"


def test_pipeline_error_handling(mock_pipeline_components):
    """
    Test that an exception in any stage raises an error and logs the event.
    """
    (parser_mock, chunker_mock, embedding_mock, classifier_mock, stripper_mock) = mock_pipeline_components

    # Let the parser succeed
    parser_mock.parse.return_value = "Text from parser"

    # Let chunker fail, for example
    chunker_mock.chunk_text.side_effect = RuntimeError("Chunking error")

    pipeline = SafeContextPipeline(config_path=None)

    with pytest.raises(RuntimeError, match="Chunking error"):
        pipeline.process_input("fake_file.txt")

    # Confirm chunker_mock was indeed called before raising
    chunker_mock.chunk_text.assert_called_once()
    # Downstream methods (embedding, classify, sanitize) should not be called
    embedding_mock.get_embeddings.assert_not_called()
    classifier_mock.classify_chunks.assert_not_called()
    stripper_mock.sanitize.assert_not_called()
