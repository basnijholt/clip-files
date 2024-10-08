# Clip Files 📋📁

![](https://media.githubusercontent.com/media/basnijholt/nijho.lt/main/content/project/clip-files/featured.jpg)

![Build](https://github.com/basnijholt/clip-files/actions/workflows/pytest.yml/badge.svg)
[![Coverage](https://img.shields.io/codecov/c/github/basnijholt/clip-files)](https://codecov.io/gh/basnijholt/clip-files)
[![GitHub](https://img.shields.io/github/stars/basnijholt/clip-files.svg?style=social)](https://github.com/basnijholt/clip-files/stargazers)
[![PyPI](https://img.shields.io/pypi/v/clip-files.svg)](https://pypi.python.org/pypi/clip-files)
[![License](https://img.shields.io/github/license/basnijholt/clip-files)](https://github.com/basnijholt/clip-files/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/clip-files)](https://pypi.python.org/pypi/clip-files)
![Open Issues](https://img.shields.io/github/issues-raw/basnijholt/clip-files)

Introducing `clip-files` - a simple yet useful command-line utility to gather files with a specific extension, format them with their full path, and copy their contents to your clipboard with a token count for GPT-4 usage!
Makes it simple to paste in your entire project into e.g., ChatGPT or Claude.ai, now that massive token limits are available.

<details><summary><b><u>[ToC]</u></b> 📚</summary>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [:star2: Features](#star2-features)
- [:books: Usage](#books-usage)
- [:hammer_and_wrench: Installation](#hammer_and_wrench-installation)
- [:bulb: Examples](#bulb-examples)
- [:file_folder: Supported File Types](#file_folder-supported-file-types)
- [:1234: Token Counting](#1234-token-counting)
- [:heart: Support and Contributions](#heart-support-and-contributions)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

</details>

## :star2: Features

* 📋 Collect files with a specific extension and copy their formatted contents to your clipboard.
* 📁 Prepend each file with `# File: full_path_here` for clarity.
* 🧮 Calculate and display the total number of tokens used, compatible with GPT-4 tokenization.
* 🔄 Easily handle and process multiple files in various directories.

## :books: Usage

To use `clip-files`, simply provide the directory to search and the file extension you want to collect:

```bash
clip-files --help
```
Shows the help message:

<!-- CODE:BASH:START -->
<!-- echo '```bash' -->
<!-- clip-files --help -->
<!-- echo '```' -->
<!-- CODE:END -->

<!-- OUTPUT:START -->
<!-- ⚠️ This content is auto-generated by `markdown-code-runner`. -->
```bash
usage: clip-files [-h] [--initial-file INITIAL_FILE]
                  [--files FILES [FILES ...]]
                  [folder] [extension]

Collect files with a specific extension or specific files, format them for
clipboard, and count tokens.

positional arguments:
  folder                The folder to search for files.
  extension             The file extension to look for (e.g., .py, .txt).

options:
  -h, --help            show this help message and exit
  --initial-file INITIAL_FILE
                        A file containing initial instructions to prepend to
                        the clipboard content. Default is an empty string.
  --files FILES [FILES ...]
                        Specific file paths to include (e.g., --files
                        path/to/file1.py path/to/file2.md). If not provided,
                        all files with the specified extension are included.
```

<!-- OUTPUT:END -->

`clip-files` will traverse the specified folder, gather files with the desired extension, format them with their paths, and copy the results to your clipboard along with the token count.

## :hammer_and_wrench: Installation

To install `clip-files`, use pip:

```bash
pip install clip-files
```

Alternatively, clone the repository:

```bash
git clone https://github.com/basnijholt/clip-files.git
cd clip-files
pip install .
```

Or download the script directly:

```bash
wget https://raw.githubusercontent.com/basnijholt/clip-files/main/clip_files.py
```
and run it using:

```bash
python clip_files.py --help
```

## :bulb: Examples

* Collect all `.py` files in the `src` directory:

```
clip-files src .py
```

* Collect `.txt` files in `documents` and count tokens:

```
clip-files documents .txt
```

For more examples and detailed usage instructions, refer to the documentation.

## :file_folder: Supported File Types

`clip-files` supports any file type as long as you provide the correct file extension. Common examples include:

- `.py` for Python files
- `.txt` for text files
- `.md` for Markdown files

## :1234: Token Counting

`clip-files` integrates with the `tiktoken` library to calculate tokens as they would be counted in GPT-4. This helps in understanding the cost and feasibility of processing the collected text with GPT-4 models.

The script will display the total token count after copying the formatted content to the clipboard.

## :heart: Support and Contributions

We welcome feedback and contributions! If you encounter any issues or have suggestions for improvements, please file an issue on our GitHub repository. Contributions via pull requests are also appreciated.

Happy collecting and clipping! 📋📁🎉
