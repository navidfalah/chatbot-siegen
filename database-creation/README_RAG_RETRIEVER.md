# Unified RAG Retriever for Chatbot Siegen

This module provides a unified retrieval function that can search across both databases:
1. **AWMF Medical Guidelines** - Contains medical guidelines and patient information
2. **MUT_ATLAS Locations** - Contains information about health and social services locations

## Features

- Unified search across both databases with a single API
- Automatic detection of query intent (whether it's about locations, guidelines, or both)
- Automatic extraction of location from queries (e.g., "in Siegen")
- Hybrid retrieval combining dense and sparse embeddings for better results
- Reranking to improve result quality
- Standardized result format for easy integration with RAG applications

## Files

- `rag_retriever.py` - Main module that exports the retrieval function
- `test_rag_retriever.py` - Simple test script to demonstrate the retriever
- `example_rag_integration.py` - Example of integrating the retriever with a RAG system

## Prerequisites

Ensure you have a running Qdrant instance with both collections populated:
- `awmf-hybrid-rerank-de` - For medical guidelines
- `locations-v2` - For locations and services

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from rag_retriever import retrieve_content

# Get relevant content from both databases
results = retrieve_content(
    query="Wo finde ich psychologische Hilfe in Siegen?",
    top_k=3,
    qdrant_host="localhost",
    qdrant_port=6333
)

# Process the results
for result in results['results']:
    if result['type'] == 'location':
        print(f"Location: {result['name']} in {result['city']}")
    else:
        print(f"Guideline: {result['title']}")
```

### Running the Test Script

```bash
python test_rag_retriever.py
```

Options:
- `--host` - Qdrant host (default: localhost)
- `--port` - Qdrant port (default: 6333)
- `--top_k` - Number of results to display (default: 5)
- `--json` - Output results in JSON format
- `--collection` - Force search in a specific collection (guideline/location/both)
- `--debug` - Enable detailed debug logging to troubleshoot issues

### RAG Integration Example

```bash
python example_rag_integration.py
```

This demonstrates how to:
1. Retrieve relevant information using the unified retriever
2. Format the retrieved content for an LLM
3. Generate a response that combines information from both databases

## Result Format

The `retrieve_content` function returns a dictionary with the following structure:

```python
{
    'query_info': {
        'original_query': str,      # Original user query
        'detected_locations': list, # List of detected locations
        'detected_topics': list,    # List of detected health topics
        'search_type': str          # 'location', 'guideline', or 'both'
    },
    'results': [                    # List of retrieved items
        {
            'content': str,         # Raw content from the document
            'type': str,            # 'location' or 'guideline'
            # For locations:
            'name': str,            # Location name
            'address': str,         # Full address
            'city': str,            # City name
            'email': str,           # Contact email
            'phone': str,           # Contact phone
            'website': str,         # Website URL
            # For guidelines:
            'title': str,           # Guideline title
            'url': str,             # Source URL
            # Common fields:
            'metadata': dict        # Full metadata including scores
        },
        # More results...
    ]
}
```

## Advanced Usage

### Forcing Collection Type

You can force the retriever to search in a specific collection:

```python
# Only search for locations
location_results = retrieve_content(
    query="Psychologische Hilfe",
    force_collection="location"
)

# Only search for guidelines
guideline_results = retrieve_content(
    query="Symptome von Diabetes",
    force_collection="guideline"
)
```

### Automatic Location Detection and Smart Ranking

The retriever uses a robust two-step approach for location-based queries:

1. **Direct City Detection**: The system contains a list of common German cities and checks if any appear in the query.

2. **Semantic Search + Smart Reranking**: Instead of strict filtering (which can be too restrictive), the system:
   - Performs a semantic search using the full query
   - Identifies results that match detected locations
   - Boosts the ranking scores for results that contain the detected locations
   - Returns the most relevant results based on both semantic relevance and location matching

This approach provides several advantages:
- No results are excluded due to overly strict filtering
- Location-relevant results are naturally prioritized
- The system works with various query formulations ("in Siegen", "Siegen", etc.)
- Multiple locations in a query are handled appropriately

### Health Topic Detection and Smart Ranking

Similar to location detection, the retriever handles health topic queries with an intelligent approach:

1. **Health Topic Detection**: The system contains a list of common German health topics and identifies them in queries.

2. **Semantic Search + Smart Reranking**: For medical guideline results:
   - Performs a semantic search using the full query
   - Identifies guidelines that contain the detected health topics
   - Boosts ranking scores based on topic relevance, with higher boosts for topics in titles
   - Returns results based on both semantic relevance and topic matching

Benefits of this approach:
- Health topic relevant results are naturally prioritized
- Guidelines mentioning the specific health topic multiple times receive higher rankings
- Topic detection works with various query formulations
- Multiple health topics in a query are handled appropriately

### Debug Mode

You can enable debug mode to help troubleshoot retrieval issues:

```python
results = retrieve_content(
    query="Wo finde ich psychologische Hilfe in Siegen?",
    debug=True
)
```

This will print detailed information about:
- How the query is processed
- What locations are detected (if any)
- What health topics are detected (if any)
- What search strategies are applied
- How results are boosted based on detected entities
- How many results are found at each step

You can also use the `--debug` flag with the test script:

```bash
python test_rag_retriever.py --debug
```
