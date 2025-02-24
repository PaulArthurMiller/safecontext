"""
Document Parser module for SafeContext library.

This module provides functionality to parse various document formats and extract raw text.
Supported formats include plain text, HTML, PDF, DOCX, and JSON.
"""

import json
import os
from pathlib import Path
from typing import Union, Optional
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document
from safecontext.config import config
from safecontext.utils.logger import SafeContextLogger

# Configure logging
logger = SafeContextLogger(__name__)

class UnsupportedFormatError(Exception):
    """Raised when attempting to parse an unsupported file format."""
    pass

class DocumentParser:
    """
    A class to handle parsing of various document formats into plain text.
    
    Supported formats:
    - Plain text (.txt)
    - PDF (.pdf)
    - HTML (.html, .htm)
    - Microsoft Word (.docx)
    - JSON (.json)
    """
    
    def __init__(self):
        self.supported_formats = ['txt', 'pdf', 'html', 'htm', 'docx', 'json']

    def parse(self, input_source: Union[str, Path], format_hint: Optional[str] = None) -> str:
        """
        Parse input source and return cleaned, normalized text.
        
        Args:
            input_source: File path or raw string content to parse
            format_hint: Optional string indicating the format (e.g., 'html', 'pdf')
        
        Returns:
            str: Cleaned and normalized text content
            
        Raises:
            UnsupportedFormatError: If the file format is not supported
            FileNotFoundError: If the input file doesn't exist
        """
        if isinstance(input_source, (str, Path)):
            input_path = Path(input_source)
            if input_path.exists():
                return self._parse_file(input_path)
            else:
                # Treat as raw string if not a valid file path
                return self._clean_text(str(input_source))
        return self._clean_text(str(input_source))

    def _parse_file(self, file_path: Path) -> str:
        """Parse a file based on its extension."""
        extension = file_path.suffix.lower()[1:]  # Remove the dot
        
        if extension not in self.supported_formats:
            raise UnsupportedFormatError(
                f"Format '{extension}' is not supported. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
        
        with open(file_path, 'rb') as file:
            if extension == 'txt':
                return self._parse_text(file)
            elif extension == 'pdf':
                return self._parse_pdf(file)
            elif extension in ('html', 'htm'):
                return self._parse_html(file)
            elif extension == 'docx':
                return self._parse_docx(file)
            elif extension == 'json':
                return self._parse_json(file)
        
        raise UnsupportedFormatError(f"No parser implemented for '{extension}'")

    def _parse_text(self, file) -> str:
        """Parse plain text files."""
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')
        return self._clean_text(content)

    def _parse_pdf(self, file) -> str:
        """Parse PDF files."""
        reader = PyPDF2.PdfReader(file)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        return self._clean_text('\n'.join(text))

    def _parse_html(self, file) -> str:
        """Parse HTML files."""
        soup = BeautifulSoup(file, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        return self._clean_text(soup.get_text())

    def _parse_docx(self, file) -> str:
        """Parse DOCX files."""
        doc = Document(file)
        return self._clean_text('\n'.join(paragraph.text for paragraph in doc.paragraphs))

    def _parse_json(self, file) -> str:
        """Parse JSON files."""
        data = json.load(file)
        if isinstance(data, str):
            return self._clean_text(data)
        return self._clean_text(json.dumps(data, ensure_ascii=False))

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove empty lines and normalize whitespace
        lines = [line.strip().rstrip('.') for line in text.splitlines() if line.strip()]
        text = ' '.join(lines)
        # Replace multiple spaces with single space and remove trailing period
        return ' '.join(text.split()).rstrip('.')

# Create a global instance for easy access
parser = DocumentParser()
