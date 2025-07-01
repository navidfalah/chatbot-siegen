# ==============================================================================
# SCRIPT 3: 03_chunk_documents.py
#
# PURPOSE:
#   - Loads the cleaned and enriched JSON files from Stage 2.
#   - Uses SemanticChunker with a sentence-transformer model to find the most
#     logically coherent breakpoints in the text.
#   - Attaches the rich parent metadata to every chunk.
#   - Saves all chunks into a single file for the final embedding stage.
#
# HOW TO USE:
#   1. Ensure Stages 1 and 2 have run successfully.
#   2. Install required libraries:
#      pip install -U langchain-experimental langchain-huggingface sentence-transformers
#   3. Run this script: python 03_chunk_documents.py
# ==============================================================================
import os
import json
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
INPUT_CLEAN_JSON_DIR = "data/clean_json"
OUTPUT_CHUNK_DIR = "data/chunked_data"
CHUNKED_FILENAME = "all_chunks.json"
# Using a powerful multilingual sentence-transformer model suitable for German.
# This will be downloaded automatically on first run.
EMBEDDING_MODEL_NAME = 'mixedbread-ai/deepset-mxbai-embed-de-large-v1'


def chunk_documents_semantically():
    """
    Loads processed JSON files and splits them using a semantic, embedding-based approach.
    """
    print("\n--- Stage 3: Starting Smart Semantic Chunking ---")

    os.makedirs(OUTPUT_CHUNK_DIR, exist_ok=True)
    all_chunks = []

    json_files = [f for f in os.listdir(
        INPUT_CLEAN_JSON_DIR) if f.endswith('.json')]
    if not json_files:
        print(
            f"No clean JSON files found in {INPUT_CLEAN_JSON_DIR}. Did Stage 2 run?")
        return

    # Initialize the HuggingFace embedding model
    print(f"Initializing embedding model for chunking: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Initialize the Semantic Chunker
    text_splitter = SemanticChunker(embeddings)

    for json_file in json_files:
        filepath = os.path.join(INPUT_CLEAN_JSON_DIR, json_file)
        print(f"Semantically chunking: {json_file}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        content = data["content_markdown"]
        doc_with_metadata = Document(
            page_content=content, metadata=data["metadata"])

        semantic_chunks = text_splitter.split_documents([doc_with_metadata])

        for chunk in semantic_chunks:
            chunk.metadata = {**data["metadata"], **chunk.metadata}

        all_chunks.extend(semantic_chunks)

    print(f"\nTotal documents processed: {len(json_files)}")
    print(f"Total semantic chunks created: {len(all_chunks)}")

    serializable_chunks = [{"page_content": doc.page_content,
                            "metadata": doc.metadata} for doc in all_chunks]

    output_path = os.path.join(OUTPUT_CHUNK_DIR, CHUNKED_FILENAME)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_chunks, f, ensure_ascii=False, indent=4)

    print(f"Successfully saved all semantic chunks to {output_path}")
    print("--- Stage 3: Smart Semantic Chunking Complete ---")


if __name__ == "__main__":
    chunk_documents_semantically()
