"""Test suite for `clip-files`."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import mock_open, patch, MagicMock

import pyperclip

import clip_files


def test_get_token_count() -> None:
    """Test the get_token_count function."""
    text = "Hello, how are you?"
    model = "gpt-4"
    token_count = clip_files.get_token_count(text, model)
    assert isinstance(token_count, int), "Token count should be an integer"
    assert token_count > 0, "Token count should be greater than 0"


def test_get_files_with_extension() -> None:
    """Test the get_files_with_extension function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some temporary files
        file1_path = os.path.join(temp_dir, "test1.py")
        file2_path = os.path.join(temp_dir, "test2.py")
        with open(file1_path, "w", encoding="utf-8") as f1:
            f1.write("print('Hello, world!')\n")
        with open(file2_path, "w", encoding="utf-8") as f2:
            f2.write("print('Another file')\n")

        file_contents, total_tokens, file_paths = clip_files.get_files_with_extension(temp_dir, ".py")

        assert len(file_contents) == 2, "Should find two .py files"  # noqa: PLR2004
        assert total_tokens > 0, "Total tokens should be greater than 0"
        assert file1_path in file_paths, "File path should be in the list"
        assert file2_path in file_paths, "File path should be in the list"
        assert file_contents[0].startswith("# File:"), "File content should start with # File:"


@patch("builtins.open", new_callable=mock_open, read_data="Initial instructions")
def test_main_with_initial_file(mock_file):
    """Test the main function with an initial file provided."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("print('Hello, world!')\n")
        
        initial_file_path = os.path.join(temp_dir, "initial.txt")
        with open(initial_file_path, "w", encoding="utf-8") as f:
            f.write("These are initial instructions.\n")

        args = [temp_dir, ".py", "--initial-file", initial_file_path]

        with patch("sys.argv", ["clip_files.py"] + args):
            clip_files.main()

        clipboard_content = pyperclip.paste()
        assert "These are initial instructions." in clipboard_content, "Initial instructions should be included"
        assert "# File:" in clipboard_content, "File content should be included"
        assert "My question is:" in clipboard_content, "Question prompt should be at the end"


def test_main_without_initial_file() -> None:
    """Test the main function without an initial file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("print('Hello, world!')\n")

        args = [temp_dir, ".py"]

        with patch("sys.argv", ["clip_files.py", *args]):
            clip_files.main()

        clipboard_content = pyperclip.paste()
        assert "Hello! Below are the code files from my project" in clipboard_content, "Default initial message should be included"
        assert "# File:" in clipboard_content, "File content should be included"
        assert "My question is:" in clipboard_content, "Question prompt should be at the end"
