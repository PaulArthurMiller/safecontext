import pytest
import spacy
from preprocess.chunker import TextChunker, TextChunk

@pytest.fixture
def chunker():
    return TextChunker()

@pytest.fixture
def sample_text():
    return """This is the first sentence. This is the second one!

This is a new paragraph. It has multiple sentences. And some more.

And this is the final paragraph."""

def test_init():
    chunker = TextChunker()
    assert chunker.nlp == spacy.load("en_core_web_sm")
    assert hasattr(chunker, 'chunk_size')
    assert hasattr(chunker, 'chunk_overlap')

def test_empty_text(chunker):
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   ") == []

def test_invalid_strategy(chunker):
    with pytest.raises(ValueError) as exc_info:
        chunker.chunk_text("Some text", strategy="invalid")
    assert "Unknown chunking strategy" in str(exc_info.value)

def test_sentence_chunking(chunker, sample_text):
    chunks = chunker.chunk_text(sample_text, strategy='sentence')
    assert len(chunks) == 7
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)
    assert chunks[0].text == "This is the first sentence."
    assert chunks[0].chunk_type == 'sentence'
    assert chunks[0].start_pos < chunks[0].end_pos

def test_paragraph_chunking(chunker, sample_text):
    chunks = chunker.chunk_text(sample_text, strategy='paragraph')
    assert len(chunks) == 3
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)
    assert "first sentence" in chunks[0].text
    assert "new paragraph" in chunks[1].text
    assert chunks[0].chunk_type == 'paragraph'

def test_fixed_size_chunking(chunker):
    text = "This is a test text that will be split into fixed-size chunks."
    chunks = chunker.chunk_text(text, strategy='fixed', max_chunk_size=20)
    assert all(len(chunk.text) <= 20 for chunk in chunks)
    assert chunks[0].chunk_type == 'fixed'
    
    # Test overlap
    text = "abcdefghijklmnopqrstuvwxyz"
    chunker.chunk_overlap = 5
    chunks = chunker.chunk_text(text, strategy='fixed', max_chunk_size=10)
    # Verify overlap exists between chunks
    for i in range(len(chunks)-1):
        assert chunks[i].text[-5:] in chunks[i+1].text

def test_merge_small_chunks(chunker):
    chunks = [
        TextChunk("Small", 0, 5, "sentence"),
        TextChunk("This is a longer chunk", 6, 26, "sentence"),
        TextChunk("Tiny", 27, 31, "sentence"),
        TextChunk("Another long one here", 32, 52, "sentence")
    ]
    
    merged = chunker.merge_small_chunks(chunks, min_size=10)
    assert len(merged) < len(chunks)
    assert all(len(chunk.text) >= 10 for chunk in merged[:-1])  # Last chunk might be smaller

def test_merge_small_chunks_empty(chunker):
    assert chunker.merge_small_chunks([], min_size=10) == []

def test_chunk_positions(chunker):
    text = "First sentence. Second sentence."
    chunks = chunker.chunk_text(text, strategy='sentence')
    
    # Verify positions are correct
    assert chunks[0].start_pos == 0
    assert chunks[0].end_pos == text.find("Second")
    assert chunks[1].start_pos == text.find("Second")
    assert chunks[1].end_pos == len(text)

def test_text_chunk_dataclass():
    chunk = TextChunk("test", 0, 4, "sentence")
    assert chunk.text == "test"
    assert chunk.start_pos == 0
    assert chunk.end_pos == 4
    assert chunk.chunk_type == "sentence"
