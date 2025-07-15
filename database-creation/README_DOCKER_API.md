# RAG Retriever API

This project provides a GPU-accelerated, FastAPI-based service for a Retrieval-Augmented Generation (RAG) system. The API queries either a medical guidelines database (AWMF) or a locations database (MUT_ATLAS) based on user intent. It uses a separate, managed subprocess for retrieval to ensure that resource-intensive model operations do not block the main API, and it includes an automatic shutdown for the worker process during periods of inactivity to conserve GPU memory.

## Features

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

The included `Dockerfile` bakes the ML models directly into the image. This increases the image size but prevents long initialization delays when the retriever worker starts for the first time.

To build the image, navigate to the project's root directory and run:

```sh
# We use --no-cache to ensure models are freshly downloaded during the build
docker build --no-cache -t rag-api:latest .
```

This process will take some time as it downloads several gigabytes of model data.

### 2. Run the Docker Container

To run the container, you need to provide the necessary environment variables. The command below is a template:

```sh
docker run \
    --gpus all \
    -d \
    -p 8000:8000 \
    -e RAG_API_KEY="your-secret-api-key" \
    -e QDRANT_HOST="host.docker.internal" \
    -e QDRANT_PORT="6333" \
    -e INACTIVITY_TIMEOUT="600" \
    --name my-rag-api \
    rag-api:latest
```

---

## Environment Variables

Configure the following variables using the `-e` flag with `docker run`:

| Variable           | Description                                                        | Example                      |
|--------------------|--------------------------------------------------------------------|------------------------------|
| `RAG_API_KEY`      | The secret API key required to access the `/retrieve` endpoint.     | siegen-chatbot-key-2025      |
| `QDRANT_HOST`      | The hostname or IP address of your running Qdrant server.           | host.docker.internal         |
| `QDRANT_PORT`      | The port your Qdrant server is listening on.                        | 6333                         |
| `INACTIVITY_TIMEOUT` | The time in seconds before the inactive retriever worker process is automatically shut down. Defaults to 600. | 300 |

> **Note on QDRANT_HOST:** If your Qdrant database is another Docker container on the same custom network, you can use its container name (e.g., `qdrant-db`). If Qdrant is running on your host machine, use `host.docker.internal` to allow the container to connect to it.

---

## How to Use the API

Once the container is running, you can interact with the API endpoints. The retriever worker process will start automatically upon the first API call.

### Health Check

To check the health of the API and the status of the retriever worker, send a GET request to the root endpoint. This will also provide information about GPU usage if a worker is active.

```sh
curl http://localhost:8000/
```

**Inactive Worker Response:**

```json
{"status":"healthy","message":"RAG Retriever API is running","worker_status":"inactive","worker_device":"N/A"}
```

**Active Worker Response:**

```json
{"status":"healthy","message":"RAG Retriever API is running","worker_status":"active","worker_device":"cuda (NVIDIA GeForce RTX 4090)"}
```

### Retrieve Content

To get results, send a POST request to the `/retrieve` endpoint. You must include your API key in the `X-API-Key` header.

```sh
curl -X POST "http://localhost:8000/retrieve" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-secret-api-key" \
     -d '{"query": "Was sind die Symptome von Depression?", "top_k": 3, "debug": false}'
```

#### Request Body Parameters

- `query` (**str**, required): The user query string.
- `top_k` (**int**, optional, default: 5): The final number of results to return after reranking.
- `collection` (**str**, optional, default: None): Force the search to a specific collection. Accepts `location`, `guideline`, or `both`. If null, intent detection is used.
- `debug` (**bool**, optional, default: False): Set to true to receive additional debug information in the server logs.