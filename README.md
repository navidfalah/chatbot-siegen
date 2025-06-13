# Chatbot Siegen

An intelligent conversational AI system that uses advanced analysis to understand and respond to user queries across multiple domains.

## Features

- **Intelligent Relevance Analysis**: Determines if questions are within the system's knowledge domains
- **Context-Aware Responses**: Maintains conversation history for better understanding
- **Multi-Domain Support**: Handles questions about programming, science, technology, and general topics
- **Adaptive Clarification**: Asks specific questions when more information is needed
- **Confidence Scoring**: Uses confidence metrics to determine response strategies

## Installation

```bash
# Clone the repository
git clone https://github.com/navidfalah/chatbot-siegen.git
cd chatbot-siegen

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## Quick Start

```python
from src.main import run_chatbot

# Start the chatbot
run_chatbot()
```

Or run directly from command line:

```bash
python -m src.main
```

## Project Structure

```
chatbot-siegen/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── gemma_client.py              # Gemma 3 API integration
│   │   ├── llm_manager.py               # LLM response handling
│   │   ├── prompt_templates.py          # System and user prompts
│   │   └── response_processor.py        # Post-process LLM responses
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py                 # Document retrieval system
│   │   ├── embeddings.py                # Text embedding generation
│   │   ├── vector_store.py              # Vector database operations
│   │   ├── document_processor.py        # Document chunking and preprocessing
│   │   ├── reranker.py                  # Retrieved document reranking
│   │   └── context_builder.py           # Build context from retrieved docs
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py                    # SQLAlchemy/Pydantic models
│   │   ├── connection.py                # Database connection management
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   ├── conversation_repo.py     # Conversation CRUD operations
│   │   │   ├── document_repo.py         # Document storage operations
│   │   │   └── user_repo.py             # User data operations
│   │   └── migrations/
│   │       ├── __init__.py
│   │       └── versions/
│   │
│   ├── conversation/
│   │   ├── __init__.py
│   │   ├── conversation_system.py       # Enhanced with RAG
│   │   ├── context_manager.py           # Conversation context handling
│   │   ├── memory.py                    # Conversation memory system
│   │   └── session_manager.py           # User session management
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── chat_models.py               # Chat-related data models
│   │   ├── rag_models.py                # RAG-specific models
│   │   ├── user_models.py               # User data models
│   │   └── document_models.py           # Document structure models
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py                  # Chat endpoints
│   │   │   ├── documents.py             # Document management endpoints
│   │   │   └── health.py                # Health check endpoints
│   │   ├── middleware.py                # API middleware
│   │   └── dependencies.py              # FastAPI dependencies
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chat_service.py              # Main chat orchestration
│   │   ├── document_service.py          # Document ingestion service
│   │   ├── embedding_service.py         # Embedding generation service
│   │   └── search_service.py            # Search and retrieval service
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── constants.py                 # Enhanced with RAG constants
│   │   ├── config.py                    # Configuration management
│   │   ├── logging.py                   # Logging configuration
│   │   ├── validators.py                # Input validation utilities
│   │   └── helpers.py                   # General helper functions
│   │
│   └── web/
│       ├── __init__.py
│       ├── static/
│       │   ├── css/
│       │   ├── js/
│       │   └── images/
│       └── templates/
│           ├── chat.html
│           └── index.html
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_ai/
│   │   ├── test_rag/
│   │   ├── test_database/
│   │   ├── test_conversation/
│   │   └── test_services/
│   ├── integration/
│   │   ├── test_api/
│   │   ├── test_rag_pipeline/
│   │   └── test_chat_flow/
│   └── fixtures/
│       ├── sample_documents/
│       └── test_data.json
│
├── data/
│   ├── documents/                       # Document storage
│   ├── embeddings/                      # Cached embeddings
│   └── logs/                           # Application logs
│
├── config/
│   ├── development.yaml
│   ├── production.yaml
│   └── testing.yaml
│
├── scripts/
│   ├── setup_database.py               # Database initialization
│   ├── ingest_documents.py             # Bulk document ingestion
│   ├── create_embeddings.py            # Batch embedding creation
│   └── migrate_data.py                 # Data migration utilities
│
├── docs/
│   ├── api.md                          # API documentation
│   ├── architecture.md                 # System architecture
│   ├── deployment.md                   # Deployment guide
│   └── rag_setup.md                    # RAG system setup guide
│
└── examples/
    ├── basic_chat.py                   # Simple chat example
    ├── rag_query.py                    # RAG query example
    ├── document_upload.py              # Document ingestion example
    └── batch_processing.py             # Batch processing example

```

## Usage Example

```python
from src.conversation.conversation_system import ConversationSystem

# Initialize the conversation system
chatbot = ConversationSystem()

# Start interactive conversation
chatbot.start_point()
```

## Configuration

You can customize the chatbot's behavior by modifying the constants in `src/utils/constants.py`:

- Knowledge domains
- Confidence thresholds
- Maximum clarification attempts
- Response templates

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Navid Falah

## Acknowledgments

- Built with Python 3.8+
- Inspired by modern conversational AI systems
