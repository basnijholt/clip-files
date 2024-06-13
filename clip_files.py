#!/usr/bin/env python3
"""clip-files: A utility to copy and format files with a specific extension for clipboard use."""

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


def get_files_with_extension(folder_path: str, file_extension: str) -> tuple[list[str], int, list[str]]:
    """Collect files with the specified extension from the folder and format their content.

    Args:
    ----
        folder_path: The folder to search for files.
        file_extension: The file extension to look for.

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
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    formatted_content = f"# File: {file_path}\n{content}"
                    file_contents.append(formatted_content)
                    total_tokens += get_token_count(formatted_content)

    return file_contents, total_tokens, file_paths


def main() -> None:
    """Main function to handle the collection, formatting, and clipboard operations.

    Parses command-line arguments, collects and formats files, and copies the result to the clipboard.
    """
    parser = argparse.ArgumentParser(
        description="Collect files with a specific extension, format them for clipboard, and count tokens.",
    )
    parser.add_argument("folder", type=str, help="The folder to search for files.")
    parser.add_argument(
        "extension",
        type=str,
        help="The file extension to look for (e.g., .py, .txt).",
    )
    parser.add_argument(
        "--initial-file",
        type=str,
        default="",
        help="A file containing initial instructions to prepend to the clipboard content. Default is an empty string.",
    )
    args = parser.parse_args()

    folder_path = args.folder
    file_extension = args.extension
    initial_file_path = args.initial_file

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return

    initial_message = ""
    if initial_file_path and os.path.isfile(initial_file_path):
        with open(initial_file_path, encoding="utf-8") as f:
            initial_message = f.read()
    else:
        initial_message = (
            "Hello! Below are the code files from my project that I need assistance with. Each file is prefixed with its path for reference.\n\n"
        )

    file_contents, total_tokens, file_paths = get_files_with_extension(folder_path, file_extension)

    if not file_contents:
        print(f"No files with extension {file_extension} found in {folder_path}.")
        return

    file_list_message = "## Files Included\n" + "\n".join([f"{i+1}. {path}" for i, path in enumerate(file_paths)])
    combined_initial_message = f"{initial_message}\n{file_list_message}\n\n"

    combined_content = combined_initial_message + "\n\n" + "\n\n".join(file_contents)

    pyperclip.copy(combined_content)
    print("The collected file contents have been copied to the clipboard.")
    print(f"Total number of tokens used: {total_tokens}")


if __name__ == "__main__":
    main()
