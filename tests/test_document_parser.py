import pytest
from pathlib import Path
import json
from io import BytesIO
from unittest.mock import mock_open, patch, MagicMock
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup

from preprocess.document_parser import DocumentParser, UnsupportedFormatError

@pytest.fixture
def parser():
    return DocumentParser()

@pytest.fixture
def sample_text():
    return "This is\n  some sample\n\ntext with  extra  spaces."

@pytest.fixture
def sample_html():
    return """
    <html>
        <head>
            <style>
                .hidden { display: none; }
            </style>
            <script>
                console.log("test");
            </script>
        </head>
        <body>
            <h1>Title</h1>
            <p>This is a paragraph.</p>
            <p>Another paragraph.</p>
        </body>
    </html>
    """

def test_init():
    parser = DocumentParser()
    assert hasattr(parser, 'supported_formats')
    assert isinstance(parser.supported_formats, (list, tuple, set))

def test_clean_text(parser, sample_text):
    cleaned = parser._clean_text(sample_text)
    assert cleaned == "This is some sample text with extra spaces"
    assert "  " not in cleaned  # No double spaces
    assert "\n\n" not in cleaned  # No double newlines

def test_parse_raw_string(parser):
    raw_text = "Raw\n  string\n\nwith spaces"
    result = parser.parse(raw_text)
    assert result == "Raw string with spaces"

def test_parse_nonexistent_file(parser):
    result = parser.parse("This is just text, not a file path")
    assert isinstance(result, str)
    assert result.strip() == "This is just text, not a file path"

def test_unsupported_format(parser):
    with pytest.raises(UnsupportedFormatError) as exc_info:
        parser._parse_file(Path("test.unsupported"))
    assert "not supported" in str(exc_info.value)

@patch("builtins.open", new_callable=mock_open, read_data="Sample text content")
def test_parse_text_file(mock_file, parser):
    result = parser._parse_file(Path("test.txt"))
    assert result == "Sample text content"
    mock_file.assert_called_once()

@patch.object(Path, "exists", return_value=True)
@patch("preprocess.document_parser.open", new_callable=mock_open, read_data=b"Fake PDF bytes")
@patch("preprocess.document_parser.PdfReader")
def test_parse_pdf(mock_pdfreader, mock_file, mock_exists, parser):
    # 1) Mock up the PdfReader instance
    mock_pdf = MagicMock()
    mock_pdf.pages = [
        MagicMock(extract_text=lambda: "Page 1 content"),
        MagicMock(extract_text=lambda: "Page 2 content"),
    ]
    mock_pdfreader.return_value = mock_pdf

    # 2) Now call your parser
    result = parser._parse_file(Path("test.pdf"))

    # 3) Verify it worked
    assert "Page 1 content" in result
    assert "Page 2 content" in result
    # And if you want to confirm open(...) was called with 'rb':
    mock_file.assert_called_once_with(Path("test.pdf"), "rb")

@patch("builtins.open", new_callable=mock_open)
def test_parse_html(mock_file, parser, sample_html):
    mock_file.return_value = BytesIO(sample_html.encode())
    result = parser._parse_file(Path("test.html"))
    
    # Check that content is preserved
    assert "Title" in result
    assert "This is a paragraph" in result
    assert "Another paragraph" in result
    
    # Check that script and style content is removed
    assert "hidden" not in result
    assert "console.log" not in result

@patch.object(Path, "exists", return_value=True)
@patch("preprocess.document_parser.open", new_callable=mock_open, read_data=b"Fake docx bytes")
@patch("preprocess.document_parser.Document")
def test_parse_docx(mock_document_class, mock_file, mock_exists, parser):
    # 1) Mock up the Document instance
    mock_doc = MagicMock()
    mock_doc.paragraphs = [
        MagicMock(text="Paragraph 1"),
        MagicMock(text="Paragraph 2"),
    ]
    mock_document_class.return_value = mock_doc

    # 2) Call your parser
    result = parser._parse_file(Path("test.docx"))

    # 3) Verify
    assert "Paragraph 1" in result
    assert "Paragraph 2" in result
    mock_file.assert_called_once_with(Path("test.docx"), "rb")

@patch("builtins.open", new_callable=mock_open)
def test_parse_json(mock_file, parser):
    # Test with string JSON
    json_str = json.dumps("Simple string content")
    mock_file.return_value = BytesIO(json_str.encode())
    result = parser._parse_file(Path("test.json"))
    assert result == "Simple string content"

    # Test with complex JSON
    json_obj = json.dumps({"key": "value", "nested": {"data": "test"}})
    mock_file.return_value = BytesIO(json_obj.encode())
    result = parser._parse_file(Path("test.json"))
    assert "key" in result
    assert "value" in result
    assert "nested" in result

def test_global_instance():
    from preprocess.document_parser import parser
    assert isinstance(parser, DocumentParser)
