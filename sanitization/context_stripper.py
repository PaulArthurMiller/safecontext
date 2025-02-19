"""
Module for sanitizing directive text by stripping or rephrasing response-solicitation parts
while preserving factual content.
"""

from dataclasses import dataclass
from typing import List, Union, Optional, Sequence
import re
from utils.logger import SafeContextLogger

# Configure logging
logger = SafeContextLogger(__name__)

@dataclass
class StripperConfig:
    """Configuration for the context stripping process."""
    
    # Threshold for considering text as partially directive
    partial_directive_threshold: float = 0.5
    
    # Whether to preserve question marks in factual questions
    preserve_questions: bool = True
    
    # List of patterns to always remove
    removal_patterns: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.removal_patterns is None:
            self.removal_patterns = [
                r"(?i)please\s+(?:help|assist|provide|tell|explain)\s*",
                r"(?i)can\s+you\s+(?:help|tell|explain|provide)\s+(?:me\s+)?(?:about\s+)?",
                r"(?i)i\s+need\s+(?:help|assistance|you\s+to)\s*",
                r"(?i)write\s+(?:a|an|the)\s*",
                r"(?i)give\s+me\s+(?:a|an|the)\s*",
                r"(?i)tell\s+(?:me\s+)?(?:about\s+)?",
            ]

class ContextStripper:
    """
    A class for sanitizing directive text by removing response-solicitation elements
    while preserving factual content.
    
    Features:
    - Handles both single chunks and lists of chunks
    - Configurable stripping behavior
    - Preserves factual content and structure
    - Special handling for partially directive text
    """
    
    def __init__(self, config: Optional[StripperConfig] = None):
        """
        Initialize the ContextStripper with optional configuration.
        
        Args:
            config: Optional StripperConfig object for customizing behavior
        """
        self.config = config or StripperConfig()
    
    def sanitize(self, 
                text: Union[str, Sequence[str]], 
                directive_score: Optional[Union[float, Sequence[float]]] = None) -> Union[str, List[str]]:
        """
        Sanitize text by removing directive elements while preserving factual content.
        
        Args:
            text: Single text chunk or list of chunks to sanitize
            directive_score: Optional classification score(s) for partial directive handling
            
        Returns:
            Sanitized text or list of sanitized texts
            
        Examples:
            >>> stripper = ContextStripper()
            >>> stripper.sanitize("Please help explain how photosynthesis works")
            "How photosynthesis works"
            >>> stripper.sanitize("Can you tell me about the French Revolution?")
            "The French Revolution"
        """
        if isinstance(text, str):
            score = directive_score if isinstance(directive_score, (int, float)) else 1.0
            return self._sanitize_chunk(text, float(score))
        
        # Handle sequence of texts
        texts = list(text)  # Convert to list to handle any sequence type
        if directive_score is None:
            scores = [1.0] * len(texts)
        elif isinstance(directive_score, (int, float)):
            scores = [float(directive_score)] * len(texts)
        else:
            scores = [float(score) for score in directive_score]
            
        return [self._sanitize_chunk(t, s) for t, s in zip(texts, scores)]
    
    def _sanitize_chunk(self, text: str, directive_score: float) -> str:
        """
        Sanitize a single text chunk based on its directive score.
        
        Args:
            text: Text chunk to sanitize
            directive_score: Classification score indicating directive confidence
            
        Returns:
            Sanitized text
        """
        # Handle empty or whitespace text
        if not text or text.isspace():
            return text
            
        # Apply removal patterns
        sanitized = text
        if self.config.removal_patterns:  # Check if not None
            for pattern in self.config.removal_patterns:
                sanitized = re.sub(pattern, '', sanitized)
            
        # Clean up whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        # Handle question marks based on config
        if not self.config.preserve_questions:
            sanitized = sanitized.replace('?', '.')
            
        # Convert imperative to declarative where possible
        sanitized = self._convert_imperative(sanitized)
        
        # Handle partially directive text differently if score is below threshold
        if directive_score < self.config.partial_directive_threshold:
            sanitized = self._handle_partial_directive(sanitized)
            
        return sanitized
    
    def _convert_imperative(self, text: str) -> str:
        """Convert imperative sentences to declarative form."""
        # Remove leading imperative verbs
        text = re.sub(r'^(?:explain|describe|discuss|outline|analyze)\s+', '', text, flags=re.I)
        
        # Convert "How to" to "How"
        text = re.sub(r'^how\s+to\s+', 'How ', text, flags=re.I)
        
        return text
    
    def _handle_partial_directive(self, text: str) -> str:
        """Special handling for text that's only partially directive."""
        # For partially directive text, still remove common directive phrases
        for pattern in [
            r"(?i)please\s+(?:help|assist|provide|tell|explain)\s*",
            r"(?i)tell\s+(?:me\s+)?(?:about\s+)?",
        ]:
            text = re.sub(pattern, '', text)
        return text.strip()
