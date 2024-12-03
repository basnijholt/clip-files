"""Test suite for `clip-files`."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import patch

import pyperclip
import pytest

import clip_files

if TYPE_CHECKING:
    from pathlib import Path


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


def test_generate_combined_content_with_initial_file(tmp_path: Path) -> None:
    """Test the generate_combined_content function with an initial file provided."""
    # Create a test Python file in the temporary directory
    file_path = tmp_path / "test.py"
    file_path.write_text("print('Hello, world!')\n", encoding="utf-8")

    # Create an initial instructions file in the temporary directory
    initial_file_path = tmp_path / "initial.txt"
    initial_file_path.write_text("These are initial instructions.\n", encoding="utf-8")

    # Call the generate_combined_content function
    combined_content, total_tokens = clip_files.generate_combined_content(
        folder_path=str(tmp_path),
        file_extension=".py",
        initial_file_path=str(initial_file_path),
    )

    # Verify the combined content includes the initial instructions
    assert "These are initial instructions." in combined_content, combined_content
    assert "# File:" in combined_content, "File content should be included"
    assert "test.py" in combined_content, "File path should be included in the combined content"
    assert "print('Hello, world!')" in combined_content, "File content should be included in the combined content"
    assert "My question is:" in combined_content, "Question prompt should be at the end"

    # Copy the combined content to clipboard for further verification
    pyperclip.copy(combined_content)
    clipboard_content = pyperclip.paste()

    assert clipboard_content == combined_content, "Clipboard content should match the combined content generated"

    # Ensure total tokens are counted correctly
    assert total_tokens > 0, "Total tokens should be a positive integer"


def test_generate_combined_content_without_initial_file(tmp_path: Path) -> None:
    """Test the generate_combined_content function without an initial file provided."""
    # Create a test Python file in the temporary directory
    file_path = tmp_path / "test.py"
    file_path.write_text("print('Hello, world!')\n", encoding="utf-8")

    # Call the generate_combined_content function
    combined_content, total_tokens = clip_files.generate_combined_content(folder_path=str(tmp_path), file_extension=".py")

    # Verify the combined content includes the default initial message
    assert "Hello! Below are the code files from my project" in combined_content, "Default initial message should be included"
    assert "# File:" in combined_content, "File content should include file path and content"
    assert "test.py" in combined_content, "File path should be included in the combined content"
    assert "print('Hello, world!')" in combined_content, "File content should be included in the combined content"
    assert "My question is:" in combined_content, "Question prompt should be at the end"

    # Copy the combined content to clipboard for further verification
    pyperclip.copy(combined_content)
    clipboard_content = pyperclip.paste()

    assert clipboard_content == combined_content, "Clipboard content should match the combined content generated"

    # Ensure total tokens are counted correctly
    assert total_tokens > 0, "Total tokens should be a positive integer"


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


def test_generate_combined_content_with_selected_files(tmp_path: Path) -> None:
    """Test the generate_combined_content function with specific files selected."""
    # Create multiple test Python files in the temporary directory
    file1_path = tmp_path / "test1.py"
    file2_path = tmp_path / "test2.py"
    file3_path = tmp_path / "test3.py"
    file1_path.write_text("print('Hello from test1')\n", encoding="utf-8")
    file2_path.write_text("print('Hello from test2')\n", encoding="utf-8")
    file3_path.write_text("print('Hello from test3')\n", encoding="utf-8")

    # Specify the selected files
    selected_files = ["test1.py", "test3.py"]

    # Call the generate_combined_content function with selected_files
    combined_content, total_tokens = clip_files.generate_combined_content(
        folder_path=str(tmp_path),
        file_extension=".py",
        selected_files=selected_files,
    )

    # Verify that only the selected files are included in the combined content
    assert "test1.py" in combined_content, "Selected file1 should be included"
    assert "test3.py" in combined_content, "Selected file3 should be included"
    assert "test2.py" not in combined_content, "Non-selected file2 should not be included"

    # Ensure total tokens reflect only the included files
    token_count_test1 = clip_files.get_token_count(f"# File: {file1_path}\nprint('Hello from test1')\n")
    token_count_test3 = clip_files.get_token_count(f"# File: {file3_path}\nprint('Hello from test3')\n")
    expected_total_tokens = (
        token_count_test1
        + token_count_test3
        + clip_files.get_token_count(
            "Hello! Below are the code files from my project that I need assistance with. Each file is prefixed with its path for reference.\n\n"
            "## Files Included\n1. " + str(file1_path) + "\n2. " + str(file3_path) + "\n\n"
            "\n\nThis was the last file in my project. My question is:",
        )
    )
    assert total_tokens == expected_total_tokens, "Total tokens should match the sum of selected files and initial messages"

    # Copy the combined content to clipboard for further verification
    pyperclip.copy(combined_content)
    clipboard_content = pyperclip.paste()

    assert clipboard_content == combined_content, "Clipboard content should match the combined content generated"


def test_invalid_directory() -> None:
    """Test generate_combined_content with an invalid directory."""
    with pytest.raises(ValueError, match="is not a valid directory"):
        clip_files.generate_combined_content("/nonexistent/path", ".py")


def test_no_matching_files() -> None:
    """Test generate_combined_content when no matching files are found."""
    with tempfile.TemporaryDirectory() as temp_dir, pytest.raises(ValueError, match="No files with extension .xyz found"):
        clip_files.generate_combined_content(temp_dir, ".xyz")


def test_no_matching_selected_files() -> None:
    """Test generate_combined_content when no matching selected files are found."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("print('test')")

        with pytest.raises(ValueError, match="No specified files with extension .py found"):
            clip_files.generate_combined_content(temp_dir, ".py", selected_files=["nonexistent.py"])


def test_generate_combined_content_with_specific_files() -> None:
    """Test generate_combined_content_with_specific_files function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        file1_path = os.path.join(temp_dir, "test1.py")
        file2_path = os.path.join(temp_dir, "test2.txt")

        with open(file1_path, "w", encoding="utf-8") as f1:
            f1.write("print('test1')")
        with open(file2_path, "w", encoding="utf-8") as f2:
            f2.write("test2 content")

        # Test with multiple files of different types
        combined_content, total_tokens = clip_files.generate_combined_content_with_specific_files([file1_path, file2_path])

        assert "test1.py" in combined_content
        assert "test2.txt" in combined_content
        assert total_tokens > 0


def test_generate_combined_content_with_specific_files_invalid_path() -> None:
    """Test generate_combined_content_with_specific_files with invalid file path."""
    with pytest.raises(ValueError, match="Specified file .* does not exist"):
        clip_files.generate_combined_content_with_specific_files(["nonexistent.py"])


def test_main_with_specific_files() -> None:
    """Test main function with --files argument."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file1_path = os.path.join(temp_dir, "test1.py")
        file2_path = os.path.join(temp_dir, "test2.py")

        with open(file1_path, "w", encoding="utf-8") as f1:
            f1.write("print('test1')")
        with open(file2_path, "w", encoding="utf-8") as f2:
            f2.write("print('test2')")

        with patch("sys.argv", ["clip_files.py", "--files", file1_path, file2_path]):
            clip_files.main()

        clipboard_content = pyperclip.paste()
        assert "test1.py" in clipboard_content
        assert "test2.py" in clipboard_content


def test_main_with_initial_file() -> None:
    """Test main function with --initial-file argument."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test Python file
        py_file = os.path.join(temp_dir, "test.py")
        with open(py_file, "w", encoding="utf-8") as f:
            f.write("print('test')")

        # Create initial file
        initial_file = os.path.join(temp_dir, "initial.txt")
        with open(initial_file, "w", encoding="utf-8") as f:
            f.write("Custom initial message")

        with patch("sys.argv", ["clip_files.py", temp_dir, ".py", "--initial-file", initial_file]):
            clip_files.main()

        clipboard_content = pyperclip.paste()
        assert "Custom initial message" in clipboard_content
        assert "test.py" in clipboard_content
