# RAG Retriever API

This project provides a GPU-accelerated, FastAPI-based service for a Retrieval-Augmented Generation (RAG) system. It queries either a medical guidelines database or a locations database based on user intent and provides reranked results.

### Features

* Unified Retrieval: Combines data from medical guidelines (AWMF) and locations (MUT_ATLAS).
* Intent Detection: Determines if a query is about a health topic, a location, or both.
* Search Methodology: Uses hybrid search (dense + sparse vectors) for initial candidate retrieval from a Qdrant vector database.
* Reranking: Uses `mxbai-rerank` model to provide relevant results.
* GPU Acceleration: Uses CUDA-enabled GPU for model inference.
* Containerization: Packaged as a Docker container.

## Deployment Guide

This guide explains how to build and run the RAG Retriever API using Docker.

### Prerequisites

Before you begin, ensure your host machine has the following installed:

1. Docker Engine: For running containers.
2. NVIDIA GPU: With the latest drivers installed.
3. NVIDIA Container Toolkit: To allow Docker containers to access the host's GPU.
4. A running Qdrant instance: The API needs a running Qdrant server to connect to. This can be another Docker container or a service on your network.

### 1. Build the Docker Image

The included `Dockerfile` bakes the ML models directly into the image to prevent initialization delay on the first API call.

To build the image, navigate to the project's root directory and run:

```
# We use --no-cache to ensure models are freshly downloaded during the build
docker build --no-cache -t rag-api:latest .
```

This process will take some time as it downloads several gigabytes of model data.

### 2. Run the Docker Container

To run the container, you need to provide three environment variables. The command below is a template:

```
docker run \
    --gpus all \
    -d \
    -p 8000:8000 \
    -e RAG_API_KEY="your-secret-api-key" \
    -e QDRANT_HOST="host.docker.internal" \
    -e QDRANT_PORT="6333" \
    --name my-rag-api \
    rag-api:latest
```

### Environment Variables

Configure the following variables using the `-e` flag:

| Variable | Description | Example |
| -------- | ----------- | ------- |
| `RAG_API_KEY` | The API key required to access the `/retrieve` endpoint. | `siegen-chatbot-key-2025` |
| `QDRANT_HOST` | The hostname or IP address of your running Qdrant server. | `host.docker.internal` |
| `QDRANT_PORT` | The port your Qdrant server is listening on. | `6333` |

Note on `QDRANT_HOST`: If your Qdrant database is another Docker container on the same custom network, you can use its container name (e.g., `qdrant-db`). If Qdrant is running on your host machine, use `host.docker.internal`.

## How to Use the API

Once the container is running, you can interact with the API.

#### Health Check

Check the health endpoint to ensure the service is running and using the GPU:

```
curl http://localhost:8000/
```

You should see a response like: `{"status":"healthy",...,"device":"Using CUDA ..."}`

#### Retrieve Content

To get results, send a `POST` request to the `/retrieve` endpoint with your API key in the headers:

```
curl -X POST "http://localhost:8000/retrieve" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-secret-api-key" \
     -d '{"query": "Was sind die Symptome von Depression?", "top_k": 3}'
```
