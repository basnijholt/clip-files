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

        file_contents, total_tokens, file_paths = clip_files.get_files_with_extension(temp_dir, file_extensions=[".py"])

        assert len(file_contents) == 2, "Should find two .py files"
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
        file_extensions=[".py"],
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
    combined_content, total_tokens = clip_files.generate_combined_content(
        folder_path=str(tmp_path),
        file_extensions=[".py"],
    )

    # Verify the combined content includes the default initial message
    assert clip_files.DEFAULT_INITIAL_MESSAGE in combined_content
    assert "# File:" in combined_content
    assert "test.py" in combined_content
    assert "print('Hello, world!')" in combined_content
    assert "My question is:" in combined_content

    # Copy the combined content to clipboard for further verification
    pyperclip.copy(combined_content)
    clipboard_content = pyperclip.paste()

    assert clipboard_content == combined_content

    # Ensure total tokens are counted correctly
    assert total_tokens > 0


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
        assert clip_files.DEFAULT_INITIAL_MESSAGE in clipboard_content
        assert "# File:" in clipboard_content
        assert "My question is:" in clipboard_content


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
        file_extensions=[".py"],
        selected_files=selected_files,
    )

    # Verify that only the selected files are included in the combined content
    assert "test1.py" in combined_content
    assert "test3.py" in combined_content
    assert "test2.py" not in combined_content

    # Ensure total tokens reflect only the included files
    token_count_test1 = clip_files.get_token_count(
        f"# File: {file1_path}\nprint('Hello from test1')\n",
    )
    token_count_test3 = clip_files.get_token_count(
        f"# File: {file3_path}\nprint('Hello from test3')\n",
    )
    expected_total_tokens = (
        token_count_test1
        + token_count_test3
        + clip_files.get_token_count(
            clip_files.DEFAULT_INITIAL_MESSAGE + "## Files Included\n1. " + str(file1_path) + "\n2. " + str(file3_path) + "\n\n"
            "\n\nThis was the last file in my project. My question is:",
        )
    )
    assert total_tokens == expected_total_tokens

    # Copy the combined content to clipboard for further verification
    pyperclip.copy(combined_content)
    clipboard_content = pyperclip.paste()

    assert clipboard_content == combined_content


def test_invalid_directory() -> None:
    """Test generate_combined_content with an invalid directory."""
    with pytest.raises(ValueError, match="is not a valid directory"):
        clip_files.generate_combined_content("/nonexistent/path", file_extensions=[".py"])


def test_no_matching_files() -> None:
    """Test generate_combined_content when no matching files are found."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        pytest.raises(
            ValueError,
            match=f"No files with extensions .xyz found in {temp_dir}.",
        ),
    ):
        clip_files.generate_combined_content(temp_dir, file_extensions=[".xyz"])


def test_no_matching_selected_files() -> None:
    """Test generate_combined_content when no matching selected files are found."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("print('test')")

        with pytest.raises(
            ValueError,
            match=f"No specified files with extensions .py found in {temp_dir}.",
        ):
            clip_files.generate_combined_content(
                temp_dir,
                [".py"],
                selected_files=["nonexistent.py"],
            )


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
        combined_content, total_tokens = clip_files.generate_combined_content_with_specific_files(
            [file1_path, file2_path],
        )

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

        with patch(
            "sys.argv",
            ["clip_files.py", temp_dir, ".py", "--initial-file", initial_file],
        ):
            clip_files.main()

        clipboard_content = pyperclip.paste()
        assert "Custom initial message" in clipboard_content
        assert "test.py" in clipboard_content


def test_main_with_short_flag_files() -> None:
    """Test main function with -f short flag."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file1_path = os.path.join(temp_dir, "test1.py")
        file2_path = os.path.join(temp_dir, "test2.py")
        with open(file1_path, "w", encoding="utf-8") as f1:
            f1.write("print('test1')")
        with open(file2_path, "w", encoding="utf-8") as f2:
            f2.write("print('test2')")

        with patch("sys.argv", ["clip_files.py", "-f", file1_path, file2_path]):
            clip_files.main()

        clipboard_content = pyperclip.paste()
        assert "test1.py" in clipboard_content
        assert "test2.py" in clipboard_content


def test_main_with_short_flag_initial_file() -> None:
    """Test main function with -i short flag."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test Python file
        py_file = os.path.join(temp_dir, "test.py")
        with open(py_file, "w", encoding="utf-8") as f:
            f.write("print('test')")

        # Create initial file
        initial_file = os.path.join(temp_dir, "initial.txt")
        with open(initial_file, "w", encoding="utf-8") as f:
            f.write("Custom initial message")

        with patch("sys.argv", ["clip_files.py", temp_dir, ".py", "-i", initial_file]):
            clip_files.main()

        clipboard_content = pyperclip.paste()
        assert "Custom initial message" in clipboard_content
        assert "test.py" in clipboard_content


def test_main_with_combined_short_flags() -> None:
    """Test main function with both -f and -i short flags."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        file1_path = os.path.join(temp_dir, "test1.py")
        file2_path = os.path.join(temp_dir, "test2.py")
        with open(file1_path, "w", encoding="utf-8") as f1:
            f1.write("print('test1')")
        with open(file2_path, "w", encoding="utf-8") as f2:
            f2.write("print('test2')")

        # Create initial file
        initial_file = os.path.join(temp_dir, "initial.txt")
        with open(initial_file, "w", encoding="utf-8") as f:
            f.write("Custom initial message for combined test")

        with patch("sys.argv", ["clip_files.py", "-f", file1_path, "-i", initial_file]):
            clip_files.main()

        clipboard_content = pyperclip.paste()
        assert "Custom initial message for combined test" in clipboard_content
        assert "test1.py" in clipboard_content
        assert "test2.py" not in clipboard_content  # Should only include the specified file


def test_generate_combined_content_with_multiple_extensions(tmp_path: Path) -> None:
    """Test the generate_combined_content function with multiple file extensions."""
    # Create test files with different extensions
    file1_path = tmp_path / "test1.py"
    file2_path = tmp_path / "test2.md"
    file3_path = tmp_path / "test3.txt"
    file4_path = tmp_path / "test4.js"  # This one shouldn't be included

    file1_path.write_text("print('Hello from Python')\n", encoding="utf-8")
    file2_path.write_text("# Hello from Markdown\n", encoding="utf-8")
    file3_path.write_text("Hello from text file\n", encoding="utf-8")
    file4_path.write_text("console.log('Not included');\n", encoding="utf-8")

    # Call with multiple extensions
    combined_content, total_tokens = clip_files.generate_combined_content(
        folder_path=str(tmp_path),
        file_extensions=[".py", ".md", ".txt"],
    )

    # Verify that only the expected files are included
    assert "test1.py" in combined_content
    assert "test2.md" in combined_content
    assert "test3.txt" in combined_content
    assert "test4.js" not in combined_content

    # Verify content is included
    assert "Hello from Python" in combined_content
    assert "Hello from Markdown" in combined_content
    assert "Hello from text file" in combined_content
    assert "Not included" not in combined_content

    # Ensure token count is positive
    assert total_tokens > 0


def test_is_hidden() -> None:
    """Test is_hidden function."""
    assert clip_files.is_hidden(".hidden_file") is True
    assert clip_files.is_hidden("normal_file") is False
    assert clip_files.is_hidden("/path/to/.hidden_dir") is True
    assert clip_files.is_hidden("/path/to/normal_dir") is False


def test_is_binary() -> None:
    """Test is_binary function."""
    with tempfile.NamedTemporaryFile(delete=False) as text_file:
        text_file.write(b"This is a text file")
        text_file_path = text_file.name

    with tempfile.NamedTemporaryFile(delete=False) as binary_file:
        binary_file.write(b"Binary\0File\0With\0Null\0Bytes")
        binary_file_path = binary_file.name

    try:
        assert clip_files.is_binary(text_file_path) is False
        assert clip_files.is_binary(binary_file_path) is True
        # Test with nonexistent file
        assert clip_files.is_binary("/path/to/nonexistent/file") is True
    finally:
        os.unlink(text_file_path)
        os.unlink(binary_file_path)


def test_maxdepth_parameter(tmp_path: Path) -> None:
    """Test maxdepth parameter in directory traversal."""
    # Create a directory structure with multiple levels
    level1 = tmp_path / "level1"
    level1.mkdir()
    level2 = level1 / "level2"
    level2.mkdir()
    level3 = level2 / "level3"
    level3.mkdir()

    # Create test files at each level
    (tmp_path / "root.py").write_text("print('root')", encoding="utf-8")
    (level1 / "level1.py").write_text("print('level1')", encoding="utf-8")
    (level2 / "level2.py").write_text("print('level2')", encoding="utf-8")
    (level3 / "level3.py").write_text("print('level3')", encoding="utf-8")

    # Test maxdepth=0 (only root directory)
    contents0, _, paths0 = clip_files.get_files_with_extension(
        folder_path=str(tmp_path),
        file_extensions=[".py"],
        maxdepth=0,
    )
    assert len(contents0) == 1
    assert "root.py" in contents0[0]
    assert "level1.py" not in "".join(contents0)

    # Test maxdepth=1 (root + level1)
    contents1, _, paths1 = clip_files.get_files_with_extension(
        folder_path=str(tmp_path),
        file_extensions=[".py"],
        maxdepth=1,
    )
    assert len(contents1) == 2
    assert any("level1.py" in content for content in contents1)
    assert not any("level2.py" in content for content in contents1)

    # Test maxdepth=2 (root + level1 + level2)
    contents2, _, paths2 = clip_files.get_files_with_extension(
        folder_path=str(tmp_path),
        file_extensions=[".py"],
        maxdepth=2,
    )
    assert len(contents2) == 3
    assert any("level2.py" in content for content in contents2)
    assert not any("level3.py" in content for content in contents2)

    # Test without maxdepth (all levels)
    contents_all, _, paths_all = clip_files.get_files_with_extension(
        folder_path=str(tmp_path),
        file_extensions=[".py"],
    )
    assert len(contents_all) == 4
    assert any("level3.py" in content for content in contents_all)


def test_hidden_files_and_directories(tmp_path: Path) -> None:
    """Test that hidden files and directories are skipped."""
    # Create visible files
    (tmp_path / "visible.py").write_text("print('visible')", encoding="utf-8")

    # Create hidden file
    (tmp_path / ".hidden.py").write_text("print('hidden')", encoding="utf-8")

    # Create visible directory with file
    visible_dir = tmp_path / "visible_dir"
    visible_dir.mkdir()
    (visible_dir / "file_in_visible_dir.py").write_text(
        "print('in visible dir')",
        encoding="utf-8",
    )

    # Create hidden directory with file
    hidden_dir = tmp_path / ".hidden_dir"
    hidden_dir.mkdir()
    (hidden_dir / "file_in_hidden_dir.py").write_text(
        "print('in hidden dir')",
        encoding="utf-8",
    )

    # Get files
    contents, _, paths = clip_files.get_files_with_extension(
        folder_path=str(tmp_path),
        file_extensions=[".py"],
    )

    # Check that only visible files are included
    assert len(contents) == 2
    assert any("visible.py" in content for content in contents)
    assert any("file_in_visible_dir.py" in content for content in contents)
    assert not any(".hidden.py" in content for content in contents)
    assert not any("file_in_hidden_dir.py" in content for content in contents)


def test_no_extensions_specified(tmp_path: Path) -> None:
    """Test behavior when no file extensions are specified."""
    # Create various file types
    (tmp_path / "text.txt").write_text("Text file content", encoding="utf-8")
    (tmp_path / "python.py").write_text("print('Python file')", encoding="utf-8")
    (tmp_path / "markdown.md").write_text("# Markdown file", encoding="utf-8")

    # Create a binary file
    with open(tmp_path / "binary.bin", "wb") as f:
        f.write(b"Binary\0File\0Content")

    # Get files with no extensions specified
    contents, _, paths = clip_files.get_files_with_extension(
        folder_path=str(tmp_path),
        file_extensions=None,
    )

    # Should include all non-binary files
    assert len(contents) == 3
    assert any("text.txt" in content for content in contents)
    assert any("python.py" in content for content in contents)
    assert any("markdown.md" in content for content in contents)
    assert not any("binary.bin" in content for content in contents)


def test_main_with_maxdepth() -> None:
    """Test main function with maxdepth parameter."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a directory structure with multiple levels
        level1 = os.path.join(temp_dir, "level1")
        os.mkdir(level1)
        level2 = os.path.join(level1, "level2")
        os.mkdir(level2)

        # Create test files at each level
        with open(os.path.join(temp_dir, "root.py"), "w", encoding="utf-8") as f:
            f.write("print('root')")
        with open(os.path.join(level1, "level1.py"), "w", encoding="utf-8") as f:
            f.write("print('level1')")
        with open(os.path.join(level2, "level2.py"), "w", encoding="utf-8") as f:
            f.write("print('level2')")

        # Test with maxdepth=1
        with patch("sys.argv", ["clip_files.py", temp_dir, ".py", "--maxdepth", "1"]):
            clip_files.main()

        clipboard_content = clip_files.pyperclip.paste()
        assert "root.py" in clipboard_content
        assert "level1.py" in clipboard_content
        assert "level2.py" not in clipboard_content


def test_main_with_no_extensions() -> None:
    """Test main function without specifying extensions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files of different types
        with open(os.path.join(temp_dir, "text.txt"), "w", encoding="utf-8") as f:
            f.write("Text file content")
        with open(os.path.join(temp_dir, "python.py"), "w", encoding="utf-8") as f:
            f.write("print('Python file')")

        # Create a binary file
        with open(os.path.join(temp_dir, "binary.bin"), "wb") as f:
            f.write(b"Binary\0File\0Content")

        # Test without specifying extensions
        with patch("sys.argv", ["clip_files.py", temp_dir]):
            clip_files.main()

        clipboard_content = clip_files.pyperclip.paste()
        assert "text.txt" in clipboard_content
        assert "python.py" in clipboard_content
        assert "binary.bin" not in clipboard_content


def test_unicode_decode_error_handling(tmp_path: Path) -> None:
    """Test handling of files that can't be decoded as UTF-8."""
    # Create a text file
    (tmp_path / "text.py").write_text("print('Text file')", encoding="utf-8")

    # Create a file with invalid UTF-8 encoding
    with open(tmp_path / "invalid_utf8.py", "wb") as f:
        f.write(b"print('Invalid UTF-8 character: \xff')")

    # Get files
    contents, _, paths = clip_files.get_files_with_extension(
        folder_path=str(tmp_path),
        file_extensions=[".py"],
    )

    # Only the valid UTF-8 file should be included
    assert len(contents) == 1
    assert "text.py" in contents[0]
    assert not any("invalid_utf8.py" in content for content in contents)
