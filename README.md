# Cohere Multimodal Embeddings Example

A Python application that uses Cohere's multi-modal embedding model to compare and find similarities between identification documents such as passports and residence cards (在留カード/zairyu card).

## Description

This project demonstrates the use of Cohere's `embed-multilingual-v3.0` model to:
1. Convert images of identification documents into vector embeddings
2. Compare these embeddings to find similar documents
3. Rank documents by similarity using dot product calculations

## Requirements

- Python 3.12 or higher
- Cohere API key

## Dependencies

- cohere>=5.13.12
- numpy>=2.2.3
- pillow>=11.1.0
- python-dotenv>=1.0.1
- ruff>=0.9.6 (for linting)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:username/cohere-multimodal-embeddings-example.git
cd cohere-multimodal-embeddings-example
```

2. Set up your Python environment and install dependencies using uv:
```bash
uv sync
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

## Configuration

1. Create a `.env` file in the project root:
```bash
cp .env.example .env
```

2. Add your Cohere API key to the `.env` file:
```
COHERE_API_KEY=your_api_key_here
```

## Usage

1. Place your image files in the `data/` directory
2. Run the script:
```bash
python test.py
```

The script will:
- Load and encode the reference images
- Convert them to embeddings using Cohere's API
- Compare a query image against the reference set
- Output similarity scores and rankings

## File Structure

```
.
├── data/                   # Directory for image files
├── test.py                # Main script
├── pyproject.toml         # Project configuration and dependencies
├── .env                   # Environment variables (not in repo)
└── .env.example          # Example environment file
```

## Development

This project uses `ruff` for linting and formatting. The configuration is defined in `pyproject.toml`.

To run the linter:
```bash
ruff check --fix .
```

To format the code:
```bash
ruff format .
```
