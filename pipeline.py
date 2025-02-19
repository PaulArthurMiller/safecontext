"""
Pipeline module for end-to-end text processing in SafeContext.

This module orchestrates the complete processing flow by integrating:
- Document parsing
- Text chunking
- Embedding generation
- Directive classification
- Context sanitization

Example usage:
    from pipeline import SafeContextPipeline
    
    pipeline = SafeContextPipeline()
    safe_text = pipeline.process_input("input.pdf")
"""

import logging
from pathlib import Path
from typing import Union, List, Optional

from preprocess.document_parser import DocumentParser
from preprocess.chunker import TextChunker
from embeddings.embedding_engine import EmbeddingEngine
from classification.directive_classifier import DirectiveClassifier
from sanitization.context_stripper import ContextStripper
from config import load_config, SafeContextConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeContextPipeline:
    """
    Main pipeline class for processing text through the SafeContext system.
    
    Handles the complete flow from raw input to sanitized output by coordinating
    all processing stages and maintaining configuration consistency.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline with optional custom configuration.
        
        Args:
            config_path: Optional path to custom configuration file
        """
        self.config = load_config(config_path)
        
        # Initialize components
        self.parser = DocumentParser()
        self.chunker = TextChunker()  # TextChunker doesn't take these parameters in constructor
        self.embedding_engine = EmbeddingEngine(
            model_name=self.config.model.model_name,  # Correct attribute name from ModelConfig
            config=EmbeddingConfig(
                model_name=self.config.model.model_name,
                cache_dir=self.config.model.cache_dir
            )
        )
        self.classifier = DirectiveClassifier(
            config=ClassifierConfig(),  # Using default config since we don't have matching fields
            embedding_dim=768  # Standard dimension for most embedding models
        )
        self.stripper = ContextStripper(
            config=StripperConfig()  # Using default config since we don't have matching fields
        )
        
        logger.info("SafeContext pipeline initialized successfully")
    
    def process_input(self, input_data: Union[str, Path]) -> str:
        """
        Process input text through the complete SafeContext pipeline.
        
        Args:
            input_data: Raw input text or path to input file
            
        Returns:
            Sanitized text safe for LLM processing
            
        Raises:
            ValueError: If input validation fails
            Exception: For various processing stage failures
        """
        try:
            # Convert input to Path if string looks like a file path
            if isinstance(input_data, str) and "/" in input_data or "\\" in input_data:
                input_data = Path(input_data)
            
            # Stage 1: Parse input
            if isinstance(input_data, Path):
                logger.info(f"Parsing input file: {input_data}")
                text = self.parser.parse(input_data)
            else:
                text = str(input_data)
            
            # Stage 2: Split into chunks
            logger.info("Chunking text")
            chunks = self.chunker.chunk_text(text)  # Correct method name
            chunk_texts = [chunk.text for chunk in chunks]
            
            # Stage 3: Generate embeddings
            logger.info("Generating embeddings")
            embeddings = self.embedding_engine.get_embeddings(chunk_texts)  # Correct method name
            
            # Stage 4: Classify chunks
            logger.info("Classifying chunks")
            classifications = [
                self.classifier.classify_chunk(embedding)
                for embedding in embeddings
            ]
            
            # Stage 5: Sanitize based on classification
            logger.info("Sanitizing content")
            sanitized_chunks = [
                self.stripper.sanitize(chunk, score.directive_score)
                for chunk, score in zip(chunk_texts, classifications)
            ]
            
            # Combine chunks
            result = " ".join(sanitized_chunks)
            
            logger.info("Processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            raise
