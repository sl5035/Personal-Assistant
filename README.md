# Personal Business Assistant

A RAG-based personal assistant for retrieving and answering questions about your business notes.

## Features

- Query your personal business notes using natural language
- Leverages Qdrant vector database for semantic search
- Uses OpenAI's GPT-4o for generating contextual responses
- Citations for information sources
- Simple command-line interface

## Prerequisites

- Python 3.8+
- OpenAI API key
- Qdrant cloud instance
- Daily business notes stored in Qdrant

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd Personal-Assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```bash
cp .env.example .env
```

4. Edit the `.env` file with your credentials:
```
OPENAI_API_KEY=your_openai_api_key
QDRANT_HOST=your_qdrant_host
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=your_collection_name
```

## Usage

Run the assistant:
```bash
python assistant.py
```

Ask questions about your business notes using natural language.

## Project Structure

- `assistant.py`: Main CLI interface
- `vector_store.py`: Qdrant vector database integration
- `llm.py`: OpenAI API integration and response generation
- `config.py`: Configuration settings

## Future Improvements

- Web interface using Vercel
- Enhanced citation formats
- Support for file uploads
- Analytics on frequently asked questions