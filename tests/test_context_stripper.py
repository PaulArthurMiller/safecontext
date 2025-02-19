"""Unit tests for the context_stripper module."""

import unittest
from sanitization.context_stripper import ContextStripper, StripperConfig

class TestContextStripper(unittest.TestCase):
    """Test cases for the ContextStripper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.stripper = ContextStripper()
        
    def test_basic_sanitization(self):
        """Test basic directive removal."""
        cases = [
            (
                "Please help explain how photosynthesis works",
                "how photosynthesis works"
            ),
            (
                "Can you tell me about the French Revolution?",
                "the French Revolution?"
            ),
            (
                "I need you to describe quantum mechanics",
                "quantum mechanics"
            ),
        ]
        
        for input_text, expected in cases:
            with self.subTest(input_text=input_text):
                result = self.stripper.sanitize(input_text)
                self.assertEqual(result.lower(), expected.lower())
                
    def test_partial_directive(self):
        """Test handling of partially directive text."""
        text = "The process of photosynthesis: please explain the light reactions"
        result = self.stripper.sanitize(text, directive_score=0.4)
        self.assertIn("photosynthesis", result)
        self.assertIn("light reactions", result)
        self.assertNotIn("please explain", result.lower())
        
    def test_batch_processing(self):
        """Test processing multiple chunks at once."""
        texts = [
            "Please explain gravity",
            "Tell me about electrons",
            "The structure of DNA"
        ]
        scores = [1.0, 0.8, 0.2]
        
        results = self.stripper.sanitize(texts, scores)
        self.assertEqual(len(results), 3)
        self.assertNotIn("please", results[0].lower())
        self.assertIn("dna", results[2].lower())
        
    def test_custom_config(self):
        """Test stripper with custom configuration."""
        config = StripperConfig(
            preserve_questions=False,
            removal_patterns=[r"(?i)custom_pattern"]
        )
        stripper = ContextStripper(config)
        
        result = stripper.sanitize("What is gravity?")
        self.assertTrue(result.endswith('.'))
        
    def test_edge_cases(self):
        """Test handling of edge cases."""
        cases = [
            ("", ""),  # Empty string
            ("   ", "   "),  # Whitespace
            ("?????", "?????"),  # Only punctuation
        ]
        
        for input_text, expected in cases:
            with self.subTest(input_text=input_text):
                result = self.stripper.sanitize(input_text)
                self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
