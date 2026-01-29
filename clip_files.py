#!/usr/bin/env python3
"""clip-files: A utility to copy and format files with a specific extension or specific files for clipboard use."""

from __future__ import annotations

import argparse
import os

import pyperclip
import tiktoken

FINAL_PROMPT = " This was the last file for this prompt. My question is:"
DEFAULT_INITIAL_MESSAGE = """\
Below are files that I need assistance with. Each file is surrounded with xml-like tags with its path for reference.

For example:
<file path="name">
CONTENT
</file path="name">)


"""


def get_token_count(text: str, model: str = "gpt-4") -> int:
    """Calculate the number of tokens in the provided text as per the specified model.

    Args:
    ----
        text: The text to be tokenized.
        model: The model to use for tokenization. Default is "gpt-4".

    Returns:
    -------
        The number of tokens in the text.

    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


def get_files_with_extension(
    folder_path: str,
    file_extensions: list[str] | None = None,
    selected_files: list[str] | None = None,
    maxdepth: int | None = None,
) -> tuple[list[str], int, list[str]]:
    """Collect files with the specified extension from the folder and format their content.

    Args:
    ----
        folder_path: The folder to search for files.
        file_extensions: The file extensions to look for.
        selected_files: Optional list of specific file names to include.
        maxdepth: Maximum depth to search in subdirectories (None means no limit).

    Returns:
    -------
        A tuple containing a list of formatted file contents, the total token count, and a list of file paths.

    """
    file_contents = []
    total_tokens = 0
    file_paths = []

    for root, dirs, files in os.walk(folder_path):
        # Handle maxdepth
        if maxdepth is not None:
            # Calculate current depth
            rel_path = os.path.relpath(root, folder_path)
            current_depth = 0 if rel_path == "." else rel_path.count(os.path.sep) + 1
            if current_depth >= maxdepth:
                # Don't traverse subdirectories
                dirs[:] = []

        # Filter out hidden directories
        dirs[:] = [d for d in dirs if not is_hidden(os.path.join(root, d))]

        for file in files:
            file_path = os.path.join(root, file)

            # Skip hidden files
            if is_hidden(file_path):
                continue

            # Check if file matches our criteria
            should_include = False

            if selected_files:
                # If we have specific files, only include those
                should_include = file in selected_files
            elif file_extensions:
                # If we have extensions, only include files with those extensions
                should_include = any(file.endswith(ext) for ext in file_extensions)
            else:
                # If no extensions specified, include all non-binary files
                should_include = not is_binary(file_path)

            if should_include:
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()
                except (UnicodeDecodeError, FileNotFoundError, OSError):
                    # Skip files that can't be decoded as UTF-8 (likely binary),
                    # broken symlinks, or other OS errors
                    continue
                else:
                    formatted_content = f"# File: {file_path}\n{content}"
                    file_contents.append(formatted_content)
                    total_tokens += get_token_count(formatted_content)
                    file_paths.append(file_path)

    return file_contents, total_tokens, file_paths


def generate_combined_content(
    folder_path: str,
    file_extensions: list[str] | None = None,
    initial_file_path: str = "",
    selected_files: list[str] | None = None,
    maxdepth: int | None = None,
) -> tuple[str, int]:
    """Generate combined content with file list, initial message, and file contents.

    Args:
    ----
        folder_path: The folder to search for files.
        file_extensions: The file extensions to look for.
        initial_file_path: Optional path to an initial file with instructions.
        selected_files: Optional list of specific file names to include.
        maxdepth: Maximum depth to search in subdirectories (None means no limit).

    Returns:
    -------
        Combined content as a single string and the total number of tokens.

    """
    if not os.path.isdir(folder_path):
        msg = f"{folder_path} is not a valid directory."
        raise ValueError(msg)

    initial_message = ""
    if initial_file_path and os.path.isfile(initial_file_path):
        with open(initial_file_path, encoding="utf-8") as f:
            initial_message = f.read()
    else:
        initial_message = DEFAULT_INITIAL_MESSAGE

    file_contents, files_tokens, file_paths = get_files_with_extension(
        folder_path,
        file_extensions,
        selected_files,
        maxdepth,
    )

    if not file_contents:
        if file_extensions:
            extensions_str = ", ".join(file_extensions)
            if selected_files:
                msg = f"No specified files with extensions {extensions_str} found in {folder_path}."
            else:
                msg = f"No files with extensions {extensions_str} found in {folder_path}."
        else:
            msg = f"No suitable files found in {folder_path}."
        raise ValueError(msg)

    file_list_message = "## Files Included\n" + "\n".join(
        [f"{i+1}. {path}" for i, path in enumerate(file_paths)],
    )
    combined_initial_message = f"{initial_message}\n{file_list_message}\n\n"

    combined_content = combined_initial_message + "\n\n".join(file_contents) + "\n\n" + FINAL_PROMPT

    # Calculate tokens for all parts
    initial_tokens = get_token_count(combined_initial_message)
    final_tokens = get_token_count(FINAL_PROMPT)

    # Total tokens include initial, file contents, and final prompt
    total_tokens = initial_tokens + files_tokens + final_tokens

    return combined_content, total_tokens


def generate_combined_content_with_specific_files(
    file_paths: list[str],
    initial_file_path: str = "",
) -> tuple[str, int]:
    """Generate combined content with specific files and optional initial message.

    Args:
    ----
        file_paths: List of specific file paths to include.
        initial_file_path: Optional path to an initial file with instructions.

    Returns:
    -------
        Combined content as a single string and the total number of tokens.

    """
    file_contents = []
    total_tokens = 0

    # Process each specified file
    for file_path in file_paths:
        if os.path.isdir(file_path):
            print(f"Skipping directory: {file_path}")
            continue

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except (UnicodeDecodeError, FileNotFoundError, OSError):
            print(f"Skipping unreadable file: {file_path}")
            continue

        formatted_content = f'<file path="{file_path}">\n{content}\n</file path="{file_path}">'
        file_contents.append(formatted_content)
        total_tokens += get_token_count(formatted_content)

    # Handle initial message
    initial_message = ""
    if initial_file_path and os.path.isfile(initial_file_path):
        with open(initial_file_path, encoding="utf-8") as f:
            initial_message = f.read()
    else:
        initial_message = DEFAULT_INITIAL_MESSAGE

    # Create file list message
    file_list_message = "## Files Included\n" + "\n".join(
        [f"{i+1}. {path}" for i, path in enumerate(file_paths)],
    )
    combined_initial_message = f"{initial_message}\n{file_list_message}\n\n"

    # Combine all parts
    combined_content = combined_initial_message + "\n\n".join(file_contents) + "\n\n" + FINAL_PROMPT

    # Calculate tokens for all parts
    initial_tokens = get_token_count(combined_initial_message)
    final_tokens = get_token_count(FINAL_PROMPT)

    # Total tokens
    total_tokens = initial_tokens + total_tokens + final_tokens

    return combined_content, total_tokens


def is_hidden(file_path: str) -> bool:
    """Check if a file or directory is hidden (starts with .)."""
    return os.path.basename(file_path).startswith(".")


def is_binary(file_path: str) -> bool:
    """Check if a file appears to be binary using a simple heuristic."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            return b"\0" in chunk  # Binary files typically contain null bytes
    except (OSError, PermissionError):
        return True  # Assume binary if we can't read the file


_DOC = """
Collect files with specific extensions or specific files, format them for clipboard, and count tokens.
There are two main ways to use clip-files:
1. Collecting all files with specific extensions in a folder:
   `clip-files FOLDER EXTENSION [EXTENSIONS ...]`
   Examples:
   - `clip-files . .py`  # all Python files in current directory
   - `clip-files . .py .md .txt`  # all Python, Markdown, and text files in current directory
   - `clip-files src .txt`  # all text files in src directory
   - `clip-files docs .md --initial-file instructions.txt`  # with custom instructions
2. Collecting specific files (can be of different types):
   `clip-files --files FILE [FILE ...]`
   Examples:
   - `clip-files --files src/*.py tests/*.py`  # using shell wildcards
   - `clip-files --files src/main.py docs/README.md`  # different file types
   - `clip-files -f src/*.py -i instructions.txt`  # with custom instructions

Options:
  -f, --files        Specify individual files to include
  -i, --initial-file A file containing initial instructions

Note: When using wildcards (e.g., *.py), your shell will expand them before passing to clip-files.
"""


def main() -> None:
    """Main function to handle the collection, formatting, and clipboard operations.

    Parses command-line arguments, collects and formats files, and copies the result to the clipboard.
    """
    parser = argparse.ArgumentParser(
        description=_DOC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Make 'folder' and 'extension' optional positional arguments
    parser.add_argument(
        "folder",
        type=str,
        nargs="?",
        help="The folder to search for files.",
    )
    parser.add_argument(
        "extension",
        type=str,
        nargs="*",
        help="The file extensions to look for (e.g., .py, .txt). If not provided, includes all non-hidden, non-binary files.",
    )
    parser.add_argument(
        "-i",
        "--initial-file",
        type=str,
        default="",
        help="A file containing initial instructions to prepend to the clipboard content. Default is an empty string.",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        default=None,
        help="Specific file paths to include (e.g., --files path/to/file1.py path/to/file2.md)."
        " If not provided, all files with the specified extensions are included.",
    )
    parser.add_argument(
        "-m",
        "--maxdepth",
        type=int,
        default=None,
        help="Maximum directory depth to traverse (default: no limit)",
    )

    args = parser.parse_args()

    # Custom validation to enforce mutual exclusivity
    if args.files is None:
        if not args.folder:
            parser.error(
                "the following argument is required: folder when --files is not used",
            )
    elif args.folder or args.extension:
        parser.error("folder and extension should not be provided when using --files")

    try:
        if args.files:
            combined_content, total_tokens = generate_combined_content_with_specific_files(
                file_paths=args.files,
                initial_file_path=args.initial_file,
            )
        else:
            combined_content, total_tokens = generate_combined_content(
                folder_path=args.folder,
                file_extensions=args.extension if args.extension else None,
                initial_file_path=args.initial_file,
                maxdepth=args.maxdepth,
            )

        pyperclip.copy(combined_content)
        print("The collected file contents have been copied to the clipboard.")
        print(f"Total number of tokens used: {total_tokens}")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
