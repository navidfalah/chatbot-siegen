# ==============================================================================
# SCRIPT 4 (UPGRADED): 04_hybrid_upload.py
#
# PURPOSE:
#   - Sets up a Qdrant collection to support hybrid search (dense + sparse).
#   - Uses HuggingFaceEmbeddings for high-quality dense embeddings.
#   - Uses FastEmbedSparse for efficient BM25-based sparse embeddings.
#   - Uploads all document chunks with both vector types.
#
# HOW TO USE:
#   1. Ensure Stages 1, 2, and 3 have run successfully.
#   2. Ensure Qdrant is running.
#   3. Install required libraries (see README.md).
#   4. Run this script: python 04_hybrid_upload.py
# ==============================================================================
import os
import json
import qdrant_client
from qdrant_client.http import models
from langchain_qdrant import Qdrant, FastEmbedSparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

# --- Configuration ---
CHUNK_FILE_PATH = "data/chunked_data/all_chunks.json"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# A top-performing German embedding model from the MTEB leaderboard
DENSE_EMBEDDING_MODEL_NAME = 'mixedbread-ai/deepset-mxbai-embed-de-large-v1'
# Model for creating sparse vectors (keyword-based)
SPARSE_EMBEDDING_MODEL_NAME = 'Qdrant/bm25'

# New collection name to reflect the hybrid setup
QDRANT_COLLECTION_NAME = "awmf-hybrid-rerank-de"
# Vector names for clarity in the collection
DENSE_VECTOR_NAME = "dense_vector"
SPARSE_VECTOR_NAME = "sparse_vector"


def embed_and_upload_hybrid():
    """
    Loads chunks, creates dense and sparse embeddings, and uploads them to Qdrant.
    """
    print("\n--- Stage 4: Starting Hybrid Embedding and Upload to Qdrant ---")

    # 1. Load pre-chunked documents
    try:
        with open(CHUNK_FILE_PATH, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
    except FileNotFoundError:
        print(
            f"ERROR: Chunk file not found at {CHUNK_FILE_PATH}. Did Stage 3 run correctly?")
        return

    documents = [Document(page_content=c['page_content'],
                          metadata=c['metadata']) for c in chunks_data]
    if not documents:
        print("No documents to embed.")
        return
    print(f"Loaded {len(documents)} document chunks to be embedded.")

    # 2. Initialize embedding models
    print(f"Initializing dense embedding model: {DENSE_EMBEDDING_MODEL_NAME}")
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=DENSE_EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},  # Use GPU for embedding
        encode_kwargs={'normalize_embeddings': True}
    )
    # The dimension of the dense vector model
    dense_vector_dim = 1024

    print(
        f"Initializing sparse embedding model: {SPARSE_EMBEDDING_MODEL_NAME}")
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL_NAME)

    # 3. Initialize Qdrant Client and create the collection
    print(
        f"Connecting to Qdrant and setting up collection: '{QDRANT_COLLECTION_NAME}'")
    client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Use client to create collection with specific dense and sparse vector configs
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=dense_vector_dim,
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        }
    )
    print("Collection created successfully.")

    # 4. Instantiate Langchain Qdrant store for adding documents
    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME
    )

    # 5. Add documents to the collection.
    # This method will automatically handle generating both dense and sparse vectors.
    print("Uploading documents with both dense and sparse vectors...")
    # Adjust batch_size as needed
    qdrant_store.add_documents(documents=documents, batch_size=64)

    print("\n--- Upload Complete! ---")
    print(
        f"Successfully populated the '{QDRANT_COLLECTION_NAME}' collection with hybrid vectors.")


if __name__ == "__main__":
    embed_and_upload_hybrid()
