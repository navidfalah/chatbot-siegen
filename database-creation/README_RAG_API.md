# RAG API for Siegen Chatbot

This API provides a secure way to access the RAG (Retrieval-Augmented Generation) system for the Siegen Chatbot project. It integrates medical guidelines from AWMF and location information from MUT_ATLAS, and provides a unified interface for retrieving relevant information.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/navidfalah/chatbot-siegen.git
cd chatbot-siegen/database-creation
```

2. Install PyTorch manually:

PyTorch needs to be installed manually based on your system configuration. Visit the official PyTorch installation page and follow the instructions:

```
https://pytorch.org/get-started/locally/
```

Select your preferences (OS, package manager, CUDA version, etc.) and use the generated command to install PyTorch.

3. Install dependencies:

```bash
pip install -r requirements-api.txt
```

4. Create a `.env` file with your API key:

```
RAG_API_KEY=your-secure-api-key
```

## Running the API Server

Start the API server using uvicorn:

```bash
uvicorn rag_api:app --host 0.0.0.0 --port 8000
```

For development with auto-reload:

```bash
uvicorn rag_api:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Health Check

```
GET /
```

Checks if the API is running.

### Retrieve Content

```
POST /retrieve
```

Retrieves content based on the query.

**Request Body:**

```json
{
  "query": "Wo finde ich psychologische Hilfe in Siegen?",
  "top_k": 5,
  "collection": "both",
  "debug": false
}
```

**Headers:**

```
X-API-Key: your-api-key
Content-Type: application/json
```

**Response:**

```json
{
  "query_info": {
    "original_query": "Wo finde ich psychologische Hilfe in Siegen?",
    "detected_locations": ["siegen"],
    "detected_topics": [],
    "search_type": "location"
  },
  "results": [
    {
      "content": "...",
      "type": "location",
      "name": "Psychologische Beratungsstelle Siegen",
      "address": "Example Street 123, 12345 Siegen",
      "city": "Siegen",
      "email": "info@example.com",
      "phone": "0123-456789",
      "website": "https://example.com",
      "metadata": {
        "rerank_score": 0.9876,
        "initial_hybrid_score": 0.8765
      }
    }
  ]
}
```

## Testing the API

Use the included test script to test the API:

```bash
python test_rag_api.py
```

Optional arguments:

- `--url`: API URL (default: http://localhost:8000)
- `--top_k`: Number of results to return (default: 5)
- `--collection`: Force search in a specific collection ("guideline", "location", or "both")
- `--json`: Output results in JSON format
- `--debug`: Enable debug mode
