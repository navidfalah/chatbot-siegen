# ==============================================================================
# SCRIPT: 06_hybrid_rerank_retrieval_locations.py
#
# PURPOSE:
#   - Retrieves and reranks data from the LOCATIONS hybrid collection.
#   - Automatically detects and filters by city if specified in the query.
#   - Displays location-specific information (name, contact, address).
#
# HOW TO USE:
#   1. Ensure Qdrant db is populated using '05_hybrid_upload_locations.py'.
#   2. Install required libraries: pip install mxbai-rerank langchain-qdrant numpy
#   3. Run the script: python 06_hybrid_rerank_retrieval_locations.py
# ==============================================================================
import argparse
import numpy as np
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient, models
from mxbai_rerank import MxbaiRerankV2

# --- Configuration ---
DENSE_EMBEDDING_MODEL_NAME = 'mixedbread-ai/deepset-mxbai-embed-de-large-v1'
SPARSE_EMBEDDING_MODEL_NAME = 'Qdrant/bm25'
RERANKER_MODEL_NAME = 'mixedbread-ai/mxbai-rerank-large-v2'

# CHANGED: Updated collection name to match the upload script
QDRANT_COLLECTION_NAME = "locations-v2"
DENSE_VECTOR_NAME = "dense_vector"
SPARSE_VECTOR_NAME = "sparse_vector"
CANDIDATE_COUNT = 25


def extract_location_from_query(query: str) -> tuple[str | None, str]:
    """
    Extracts a location specified with 'in [Location]' from the query.

    Returns:
        A tuple containing the extracted location (or None) and the cleaned query.
    """
    # Regex to find patterns like "in Berlin", "in siegen", "in Hamburg-Harburg"
    match = re.search(r'\s+in\s+([\w\s-]+)', query, re.IGNORECASE)
    if match:
        location = match.group(1).strip()
        # Remove the 'in [Location]' part from the query for a cleaner search
        cleaned_query = query[:match.start()] + query[match.end():]
        print(f"Found location filter: '{location}'. Applying filter.")
        return location, cleaned_query.strip()
    return None, query


def main():
    """Main function to run the interactive retrieval and evaluation test."""
    parser = argparse.ArgumentParser(
        description="Query the location vector database with hybrid search and reranking.")
    parser.add_argument("--host", type=str,
                        default="localhost", help="Qdrant instance host.")
    parser.add_argument("--port", type=int, default=6333,
                        help="Qdrant instance port.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of final results to display after reranking.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for the reranker model to manage memory.")

    args = parser.parse_args()

    # 1. Initialize models and clients
    print(
        f"Initializing dense embedding model: '{DENSE_EMBEDDING_MODEL_NAME}'...")
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=DENSE_EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print(
        f"Initializing sparse embedding model: '{SPARSE_EMBEDDING_MODEL_NAME}'...")
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL_NAME)

    print(f"Connecting to Qdrant at {args.host}:{args.port}...")
    qdrant_client = QdrantClient(host=args.host, port=args.port)

    # 2. Create the vector store object
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )

    print(f"Initializing reranker: '{RERANKER_MODEL_NAME}'...")
    reranker = MxbaiRerankV2(RERANKER_MODEL_NAME)

    # --- Interactive Query Loop ---
    while True:
        print("\n" + "=" * 50)
        query = input(
            "Enter your query (e.g., 'psychologische hilfe in Siegen' or 'exit'): ")
        if query.lower() == 'exit':
            break

        # NEW: Automatically detect location in query and create a filter
        location_filter = None
        location, cleaned_query = extract_location_from_query(query)
        if location:
            # FIXED: Use MatchText for flexible, partial matching on city names
            # instead of MatchValue which requires an exact match. This allows
            # "Frankfurt" to match "Frankfurt am Main".
            location_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="address.city",
                        match=models.MatchText(text=location)
                    )
                ]
            )

        # 3. Perform initial hybrid search with the optional location filter
        print(
            f"\nStep 1: Retrieving {CANDIDATE_COUNT} candidates with hybrid search...")
        candidate_results = vector_store.similarity_search_with_score(
            cleaned_query, k=CANDIDATE_COUNT, filter=location_filter
        )

        if not candidate_results:
            print("No results found.")
            continue

        candidate_docs = [res[0] for res in candidate_results]
        initial_scores = {doc.page_content: res[1] for doc, res in zip(
            candidate_docs, candidate_results)}

        # 4. Rerank the candidates
        print(f"Step 2: Reranking {len(candidate_docs)} candidates...")
        doc_texts = [doc.page_content for doc in candidate_docs]
        reranked_results = reranker.rank(
            query=cleaned_query,
            documents=doc_texts,
            return_documents=False,
            top_k=args.top_k,
            batch_size=args.batch_size
        )

        # 5. Combine original docs with new rerank scores
        final_documents = []
        for result in reranked_results:
            original_doc = candidate_docs[result.index]
            original_doc.metadata['rerank_score'] = result.score
            original_doc.metadata['initial_hybrid_score'] = initial_scores.get(
                original_doc.page_content, 'N/A')
            final_documents.append(original_doc)

        # 6. Display results (CHANGED to show location data)
        print(
            f"\n--- Top {len(final_documents)} Reranked Documents for '{query}' ---")
        for i, doc in enumerate(final_documents):
            rerank_score = doc.metadata.get('rerank_score', 'N/A')
            hybrid_score = doc.metadata.get('initial_hybrid_score', 'N/A')

            # Access the nested metadata from the payload
            meta = doc.metadata
            address = meta.get('address', {})
            contact = meta.get('contact', {})

            print(
                f"\n--- Result {i + 1} (Rerank Score: {rerank_score:.4f} | Initial Score: {hybrid_score:.4f}) ---")
            print(f"Name: {meta.get('name', 'N/A')}")
            print(
                f"Address: {address.get('street', '')} {address.get('house_number', '')}, {address.get('zip_code', '')} {address.get('city', 'N/A')}")
            print(f"Email: {contact.get('email', 'N/A')}")
            print(f"Phone: {contact.get('phone', 'N/A')}")
            print(f"Homepage: {contact.get('homepage', 'N/A')}")
            print("\nRetrieved Context:")
            print(doc.page_content)
            print("-" * 30)


if __name__ == "__main__":
    main()
