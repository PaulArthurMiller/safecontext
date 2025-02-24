"""
Text chunking module for SafeContext library.

This module provides functionality to split text into meaningful chunks
using various strategies including sentence-based, paragraph-based,
and semantic unit-based segmentation.
"""

from dataclasses import dataclass
from typing import List, Optional, Iterator
import spacy
from safecontext.config import config
from utils.logger import SafeContextLogger

# Configure logging
logger = SafeContextLogger(__name__)

@dataclass
class TextChunk:
    """
    Represents a chunk of text with metadata.
    
    Attributes:
        text: The actual text content of the chunk
        start_pos: Starting character position in original text
        end_pos: Ending character position in original text
        chunk_type: Type of chunk (e.g., 'sentence', 'paragraph')
    """
    text: str
    start_pos: int
    end_pos: int
    chunk_type: str

class TextChunker:
    """
    A class to handle text chunking using various strategies.
    
    Supports sentence-based, paragraph-based, and fixed-size chunking
    with configurable overlap between chunks.
    """
    
    def __init__(self):
        """Initialize the chunker with spaCy language model."""
        self.nlp = spacy.load("en_core_web_sm")
        self.chunk_size = config.preprocess.chunk_size
        self.chunk_overlap = config.preprocess.chunk_overlap

    def chunk_text(self, 
                  text: str, 
                  strategy: str = 'sentence',
                  max_chunk_size: Optional[int] = None) -> List[TextChunk]:
        """
        Split text into chunks using the specified strategy.
        
        Args:
            text: Input text to be chunked
            strategy: Chunking strategy ('sentence', 'paragraph', or 'fixed')
            max_chunk_size: Optional maximum size for chunks
            
        Returns:
            List of TextChunk objects
            
        Raises:
            ValueError: If an invalid strategy is specified
        """
        if not text.strip():
            return []

        if strategy == 'sentence':
            return list(self._sentence_chunks(text))
        elif strategy == 'paragraph':
            return list(self._paragraph_chunks(text))
        elif strategy == 'fixed':
            return list(self._fixed_size_chunks(text, max_chunk_size or self.chunk_size))
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _sentence_chunks(self, text: str) -> Iterator[TextChunk]:
        """Generate sentence-based chunks using spaCy."""
        doc = self.nlp(text)
        
        for sent in doc.sents:
            yield TextChunk(
                text=sent.text.strip(),
                start_pos=sent.start_char,
                end_pos=sent.end_char,
                chunk_type='sentence'
            )

    def _paragraph_chunks(self, text: str) -> Iterator[TextChunk]:
        """Generate paragraph-based chunks."""
        current_pos = 0
        
        # Split on double newlines to separate paragraphs
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Find the actual start position in original text
            start_pos = text.find(para, current_pos)
            end_pos = start_pos + len(para)
            current_pos = end_pos
            
            yield TextChunk(
                text=para,
                start_pos=start_pos,
                end_pos=end_pos,
                chunk_type='paragraph'
            )

    def _fixed_size_chunks(self, text: str, chunk_size: int) -> Iterator[TextChunk]:
        """Generate fixed-size chunks with optional overlap."""
        text_length = len(text)
        
        for i in range(0, text_length, chunk_size - self.chunk_overlap):
            chunk_end = min(i + chunk_size, text_length)
            chunk_text = text[i:chunk_end].strip()
            
            if chunk_text:
                yield TextChunk(
                    text=chunk_text,
                    start_pos=i,
                    end_pos=chunk_end,
                    chunk_type='fixed'
                )
            
            if chunk_end == text_length:
                break

    def merge_small_chunks(self, 
                          chunks: List[TextChunk], 
                          min_size: int) -> List[TextChunk]:
        """
        Merge chunks smaller than min_size with neighboring chunks.
        
        Args:
            chunks: List of TextChunk objects
            min_size: Minimum chunk size in characters
            
        Returns:
            List of merged TextChunk objects
        """
        if not chunks:
            return []

        merged = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            if len(current_chunk.text) < min_size:
                # Merge with next chunk
                current_chunk = TextChunk(
                    text=current_chunk.text + ' ' + next_chunk.text,
                    start_pos=current_chunk.start_pos,
                    end_pos=next_chunk.end_pos,
                    chunk_type=current_chunk.chunk_type
                )
            else:
                merged.append(current_chunk)
                current_chunk = next_chunk

        merged.append(current_chunk)
        return merged

# Create a global instance for easy access
chunker = TextChunker()
