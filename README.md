# RAG Research

## Overview

RAG Research is a Retrieval-Augmented Generation (RAG) project.

You can put files in /documents to be able to reference them in your queries.

The system uses the OpenAI API for generating embeddings and responses and the LlamaIndex for document retrieval.

### Querying the System

- Enter a query when prompted.
- Type `exit` to quit.
- Use `file:<filename>` to load a query from a file.
- Press **Enter** to use the default prompt file (`prompts/prompt.md`).

## Features

- **Streaming responses**: Interactive response generation
- **File-based prompt input**: Query using structured prompts
- **Cached embeddings**: Reduce redundant API calls
- **Document chunking**: Handles large text inputs efficiently
- **Similarity search**: Finds the most relevant documents

## Project Structure

```
.
├── README.md
├── documents
│   ├── document-you-want-to-be-able-to-reference.md
│   └── another-document-you-want-to-be-able-to-reference.md
├── prompts
│   └── prompt.md
└── src
    └── main.py  <-- Run this
```

## Installation

### Prerequisites

- Python 3.12+
- Poetry for dependency management

### Setup

Clone the repository and install dependencies using Poetry:

```sh
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-research.git
cd rag-research

# Install dependencies
poetry install
```

## Usage

### Running the System

To run the RAG system:

```sh
export OPENAI_API_KEY="your-api-key-here"
poetry run python src/main.py
```

## Configuration

Modify `pyproject.toml` for dependency management.

## Development

### Linting & Formatting

```sh
poetry run ruff check
poetry run ruff format
```

### Running Tests

```sh
poetry run pytest
```

## License

This project is licensed under the MIT License.

