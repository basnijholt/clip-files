#!/usr/bin/env python3
"""clip-files: A utility to copy and format files with a specific extension or specific files for clipboard use."""

from __future__ import annotations

import argparse
import os

import pyperclip
import tiktoken


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


def get_files_with_extension(folder_path: str, file_extension: str, selected_files: list[str] | None = None) -> tuple[list[str], int, list[str]]:
    """Collect files with the specified extension from the folder and format their content.

    Args:
    ----
        folder_path: The folder to search for files.
        file_extension: The file extension to look for.
        selected_files: Optional list of specific file names to include.

    Returns:
    -------
        A tuple containing a list of formatted file contents, the total token count, and a list of file paths.

    """
    file_contents = []
    total_tokens = 0
    file_paths = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(file_extension):
                if selected_files and file not in selected_files:
                    continue  # Skip files not in the selected list
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    formatted_content = f"# File: {file_path}\n{content}"
                    file_contents.append(formatted_content)
                    total_tokens += get_token_count(formatted_content)

    return file_contents, total_tokens, file_paths


def generate_combined_content(
    folder_path: str,
    file_extension: str,
    initial_file_path: str = "",
    selected_files: list[str] | None = None,
) -> tuple[str, int]:
    """Generate combined content with file list, initial message, and file contents.

    Args:
    ----
        folder_path: The folder to search for files.
        file_extension: The file extension to look for.
        initial_file_path: Optional path to an initial file with instructions.
        selected_files: Optional list of specific file names to include.

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
        initial_message = (
            "Hello! Below are the code files from my project that I need assistance with. Each file is prefixed with its path for reference.\n\n"
        )

    file_contents, files_tokens, file_paths = get_files_with_extension(
        folder_path,
        file_extension,
        selected_files,
    )

    if not file_contents:
        if selected_files:
            msg = f"No specified files with extension {file_extension} found in {folder_path}."
        else:
            msg = f"No files with extension {file_extension} found in {folder_path}."
        raise ValueError(msg)

    file_list_message = "## Files Included\n" + "\n".join(
        [f"{i+1}. {path}" for i, path in enumerate(file_paths)],
    )
    combined_initial_message = f"{initial_message}\n{file_list_message}\n\n"

    final_prompt = " This was the last file in my project. My question is:"

    combined_content = combined_initial_message + "\n\n".join(file_contents) + "\n\n" + final_prompt

    # Calculate tokens for all parts
    initial_tokens = get_token_count(combined_initial_message)
    final_tokens = get_token_count(final_prompt)

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
        if not os.path.isfile(file_path):
            msg = f"Specified file '{file_path}' does not exist."
            raise ValueError(msg)

        with open(file_path, encoding="utf-8") as f:
            content = f.read()
            formatted_content = f"# File: {file_path}\n{content}"
            file_contents.append(formatted_content)
            total_tokens += get_token_count(formatted_content)

    # Handle initial message
    initial_message = ""
    if initial_file_path and os.path.isfile(initial_file_path):
        with open(initial_file_path, encoding="utf-8") as f:
            initial_message = f.read()
    else:
        initial_message = (
            "Hello! Below are the code files from my project that I need assistance with. Each file is prefixed with its path for reference.\n\n"
        )

    # Create file list message
    file_list_message = "## Files Included\n" + "\n".join(
        [f"{i+1}. {path}" for i, path in enumerate(file_paths)],
    )
    combined_initial_message = f"{initial_message}\n{file_list_message}\n\n"

    # Final prompt
    final_prompt = " This was the last file in my project. My question is:"

    # Combine all parts
    combined_content = combined_initial_message + "\n\n".join(file_contents) + "\n\n" + final_prompt

    # Calculate tokens for all parts
    initial_tokens = get_token_count(combined_initial_message)
    final_tokens = get_token_count(final_prompt)

    # Total tokens
    total_tokens = initial_tokens + total_tokens + final_tokens

    return combined_content, total_tokens


def main() -> None:
    """Main function to handle the collection, formatting, and clipboard operations.

    Parses command-line arguments, collects and formats files, and copies the result to the clipboard.
    """
    parser = argparse.ArgumentParser(
        description="Collect files with a specific extension or specific files, format them for clipboard, and count tokens.",
    )
    # Make 'folder' and 'extension' optional positional arguments
    parser.add_argument("folder", type=str, nargs="?", help="The folder to search for files.")
    parser.add_argument(
        "extension",
        type=str,
        nargs="?",
        help="The file extension to look for (e.g., .py, .txt).",
    )
    parser.add_argument(
        "--initial-file",
        type=str,
        default="",
        help="A file containing initial instructions to prepend to the clipboard content. Default is an empty string.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Specific file paths to include (e.g., --files path/to/file1.py path/to/file2.md)."
        " If not provided, all files with the specified extension are included.",
    )
    args = parser.parse_args()

    # Custom validation to enforce mutual exclusivity
    if args.files is None:
        if not args.folder or not args.extension:
            parser.error("the following arguments are required: folder and extension when --files is not used")
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
                file_extension=args.extension,
                initial_file_path=args.initial_file,
            )
        pyperclip.copy(combined_content)
        print("The collected file contents have been copied to the clipboard.")
        print(f"Total number of tokens used: {total_tokens}")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
