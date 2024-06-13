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
        text (str): The text to be tokenized.
        model (str): The model to use for tokenization. Default is "gpt-4".

    Returns:
    -------
        int: The number of tokens in the text.

    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


def get_files_with_extension(folder_path: str, file_extension: str) -> tuple[list[str], int]:
    """Collect files with the specified extension from the folder and format their content.

    Args:
    ----
        folder_path (str): The folder to search for files.
        file_extension (str): The file extension to look for.

    Returns:
    -------
        tuple[list[str], int]: A tuple containing a list of formatted file contents and the total token count.

    """
    file_contents = []
    total_tokens = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    formatted_content = f"# File: {file_path}\n{content}"
                    file_contents.append(formatted_content)
                    total_tokens += get_token_count(formatted_content)
    return file_contents, total_tokens


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
        default=".clip-files",
        help="A file containing initial instructions to prepend to the clipboard content. Default is .clip-files in the current working directory.",
    )
    args = parser.parse_args()

    folder_path = args.folder
    file_extension = args.extension
    initial_file_path = args.initial_file

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return

    initial_message = ""
    if os.path.isfile(initial_file_path):
        with open(initial_file_path, encoding="utf-8") as f:
            initial_message = f.read()

    file_contents, total_tokens = get_files_with_extension(folder_path, file_extension)

    if not file_contents:
        print(f"No files with extension {file_extension} found in {folder_path}.")
        return

    combined_content = initial_message + "\n\n" + "\n\n".join(file_contents)

    pyperclip.copy(combined_content)
    print("The collected file contents have been copied to the clipboard.")
    print(f"Total number of tokens used: {total_tokens}")


if __name__ == "__main__":
    main()
