#!/usr/bin/env python3
"""clip-files: A to copy ."""

from __future__ import annotations
import os
import argparse
import pyperclip
import tiktoken


def get_token_count(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


def get_files_with_extension(folder_path, file_extension):
    file_contents = []
    total_tokens = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    formatted_content = f"# File: {file_path}\n{content}"
                    file_contents.append(formatted_content)
                    total_tokens += get_token_count(formatted_content)
    return file_contents, total_tokens


def main():
    parser = argparse.ArgumentParser(description="Collect files with a specific extension, format them for clipboard, and count tokens.")
    parser.add_argument("folder", type=str, help="The folder to search for files.")
    parser.add_argument("extension", type=str, help="The file extension to look for (e.g., .py, .txt).")
    args = parser.parse_args()

    folder_path = args.folder
    file_extension = args.extension

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return

    file_contents, total_tokens = get_files_with_extension(folder_path, file_extension)

    if not file_contents:
        print(f"No files with extension {file_extension} found in {folder_path}.")
        return

    combined_content = "\n\n".join(file_contents)

    pyperclip.copy(combined_content)
    print("The collected file contents have been copied to the clipboard.")
    print(f"Total number of tokens used: {total_tokens}")


if __name__ == "__main__":
    main()
