# Personal Assistant Project Guidelines

## Commands
- **Run application**: `python assistant.py`
- **Install dependencies**: `pip install -r requirements.txt`
- **Test**: No formal test suite currently available
- **Environment setup**: Copy `.env.example` to `.env` and fill in credentials

## Code Style

### Python Conventions
- **Imports**: Group standard library, third-party, and local imports
- **Type Hints**: Use type annotations for function parameters and return values
- **Docstrings**: Use descriptive docstrings for all functions and classes
- **Error Handling**: Use try-except blocks for external services (currently commented out)
- **Naming**: snake_case for functions/variables, CamelCase for classes

### Project Structure
- **Modular Design**: Each file has a specific purpose (vector_store.py, llm.py, etc.)
- **Configuration**: Environment variables loaded via dotenv in config.py
- **Dependencies**: LangChain for LLM interactions, Qdrant for vector storage

### Security Notes
- **API Keys**: Store in .env file (never commit to repository)
- **Sensitive Data**: Be careful with hardcoded credentials (see upsert.py)

### TODOs
- Implement proper error handling
- Add a formal testing framework
- Create CI/CD pipeline