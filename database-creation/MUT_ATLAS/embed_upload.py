# ==============================================================================
# SCRIPT: 05_hybrid_upload_locations.py (v3 - Corrected Indexing)
#
# PURPOSE:
#   - FIX: Creates a 'text' index for the city field to allow for flexible,
#     partial-text matching (e.g., "Frankfurt" can find "Frankfurt am Main").
#   - Reads data from the 'transformed_for_qdrant.json' file.
#   - Uses the 'document_to_embed' field for generating embeddings.
#   - Stores the rich 'payload' object as the metadata for each entry.
#
# HOW TO USE:
#   1. Ensure you have run the transformation script to create 'transformed_for_qdrant.json'.
#   2. Ensure Qdrant is running.
#   3. IMPORTANT: If you ran a previous version, delete the 'locations-v2' collection in Qdrant first.
#   4. Run this script to re-upload the data with the correct index.
# ==============================================================================
import os
import json
import qdrant_client
from qdrant_client.http import models
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore
from qdrant_client import QdrantClient

# --- Configuration ---
TRANSFORMED_DATA_PATH = "transformed_for_qdrant.json"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

DENSE_EMBEDDING_MODEL_NAME = 'mixedbread-ai/deepset-mxbai-embed-de-large-v1'
SPARSE_EMBEDDING_MODEL_NAME = 'Qdrant/bm25'

QDRANT_COLLECTION_NAME = "locations-v2"
DENSE_VECTOR_NAME = "dense_vector"
SPARSE_VECTOR_NAME = "sparse_vector"


def load_transformed_documents(file_path: str) -> list[Document]:
    """
    Loads the transformed location data and converts it into Langchain Documents.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            transformed_data = json.load(f)
    except FileNotFoundError:
        print(
            f"ERROR: File not found at {file_path}. Please run the transformation script.")
        return []

    documents = [
        Document(page_content=item['document_to_embed'],
                 metadata=item['payload'])
        for item in transformed_data
    ]

    if not documents:
        print("No documents loaded.")
    else:
        print(f"Loaded {len(documents)} location documents.")
    return documents


def embed_and_upload_locations():
    """
    Loads location data, creates a proper text index, and uploads embeddings to Qdrant.
    """
    print("\n--- Starting Hybrid Embedding and Upload for Locations Data ---")

    documents = load_transformed_documents(TRANSFORMED_DATA_PATH)
    if not documents:
        return

    print(f"Initializing dense embedding model: {DENSE_EMBEDDING_MODEL_NAME}")
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=DENSE_EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    dense_vector_dim = 1024

    print(
        f"Initializing sparse embedding model: {SPARSE_EMBEDDING_MODEL_NAME}")
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL_NAME)

    print(
        f"Connecting to Qdrant and setting up collection: '{QDRANT_COLLECTION_NAME}'")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
        print(
            f"Collection '{QDRANT_COLLECTION_NAME}' already exists. Deleting it to apply new index settings.")
        client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)

    print(f"Creating new collection: '{QDRANT_COLLECTION_NAME}'")
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=dense_vector_dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False))
        }
    )

    # FIXED: This is the critical change. We are creating a 'text' index which
    # allows for flexible text searching, fixing the filtering problem.
    print("Creating TEXT payload index for 'address.city' to enable flexible filtering.")
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="address.city",
        field_schema=models.TextIndexParams(
            type="text",
            tokenizer=models.TokenizerType.WHITESPACE,
            lowercase=True
        )
    )

    print("Collection and indexes created successfully.")

    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME
    )

    print("Uploading documents with both dense and sparse vectors...")
    qdrant_store.add_documents(documents=documents, batch_size=32)

    print("\n--- Upload Complete! ---")
    print(f"Successfully populated the '{QDRANT_COLLECTION_NAME}' collection.")


if __name__ == "__main__":
    embed_and_upload_locations()
