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
from embeddings.embedding_engine import EmbeddingEngine, EmbeddingConfig
from classification.directive_classifier import DirectiveClassifier, ClassifierConfig
from sanitization.context_stripper import ContextStripper, StripperConfig
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
        self.chunker = TextChunker()
        
        # Initialize embedding engine with proper config
        embedding_config = EmbeddingConfig(
            model_name=self.config.model.model_name,
            batch_size=self.config.model.batch_size,
            cache_dir=self.config.model.cache_dir
        )
        self.embedding_engine = EmbeddingEngine(
            model_name=self.config.model.model_name,
            config=embedding_config
        )
        
        # Initialize classifier with proper config
        classifier_config = ClassifierConfig(
            confidence_threshold=self.config.classification.confidence_threshold,
            max_context_length=self.config.classification.max_context_length,
            min_directive_length=self.config.classification.min_directive_length
        )
        self.classifier = DirectiveClassifier(
            config=classifier_config,
            embedding_dim=768  # Standard dimension for most embedding models
        )
        
        # Initialize context stripper with proper config
        self.stripper = ContextStripper()  # Uses default config
        
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
            sanitized_chunks = []
            for chunk, classification in zip(chunk_texts, classifications):
                # Extract directive_score from classification result
                directive_score = classification.confidence if classification.label == "directive" else 0.0
                sanitized_chunks.append(self.stripper.sanitize(chunk, directive_score))
            
            # Combine chunks
            result = " ".join(sanitized_chunks)
            
            logger.info("Processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            raise
