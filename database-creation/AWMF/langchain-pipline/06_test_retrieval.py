# ==============================================================================
# SCRIPT 6 (UPGRADED V9): 06_hybrid_rerank_retrieval.py
#
# PURPOSE:
#   - Implements a hybrid search + rerank pipeline.
#   - Captures and displays both initial hybrid scores and final rerank scores.
#   - FIX: Added the Guideline URL to the output for better context.
#
# HOW TO USE:
#   1. Ensure Qdrant db is populated using '04_hybrid_upload.py'.
#   2. Install required libraries: pip install mxbai-rerank langchain-qdrant numpy
#   3. Run the script. It will now loop, allowing multiple queries per session.
#      python 06_hybrid_rerank_retrieval.py
# ==============================================================================
import argparse
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from mxbai_rerank import MxbaiRerankV2

# --- Configuration ---
DENSE_EMBEDDING_MODEL_NAME = 'mixedbread-ai/deepset-mxbai-embed-de-large-v1'
SPARSE_EMBEDDING_MODEL_NAME = 'Qdrant/bm25'
RERANKER_MODEL_NAME = 'mixedbread-ai/mxbai-rerank-large-v2'
QDRANT_COLLECTION_NAME = "awmf-hybrid-rerank-de"
DENSE_VECTOR_NAME = "dense_vector"
SPARSE_VECTOR_NAME = "sparse_vector"
CANDIDATE_COUNT = 25


def main():
    """Main function to run the interactive retrieval and evaluation test."""
    parser = argparse.ArgumentParser(
        description="Query the vector database with hybrid search, reranking, and evaluation.")
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
        print("\n" + "="*50)
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        # 3. Perform initial hybrid search to get candidate documents WITH scores
        print(
            f"\nStep 1: Retrieving {CANDIDATE_COUNT} candidates with hybrid search...")
        # Use similarity_search_with_score to get the initial hybrid scores
        candidate_results = vector_store.similarity_search_with_score(
            query, k=CANDIDATE_COUNT
        )

        if not candidate_results:
            print("No results found from initial search.")
            continue

        # Separate documents and their initial scores
        candidate_docs = [res[0] for res in candidate_results]
        initial_scores = {doc.page_content: res[1] for doc, res in zip(
            candidate_docs, candidate_results)}

        # 4. Manually rerank the candidates
        print(
            f"Step 2: Reranking {len(candidate_docs)} candidates using batch size {args.batch_size}...")

        doc_texts = [doc.page_content for doc in candidate_docs]

        reranked_results = reranker.rank(
            query=query,
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
            # Add the initial hybrid score for comparison
            original_doc.metadata['initial_hybrid_score'] = initial_scores.get(
                original_doc.page_content, 'N/A')
            final_documents.append(original_doc)

        # 6. Display results
        print(f"\n--- Top {len(final_documents)} Reranked Documents ---")
        for i, doc in enumerate(final_documents):
            rerank_score = doc.metadata.get('rerank_score', 'N/A')
            hybrid_score = doc.metadata.get('initial_hybrid_score', 'N/A')
            original_metadata = doc.metadata

            print(
                f"\n--- Result {i+1} (Rerank Score: {rerank_score:.4f} | Initial Score: {hybrid_score:.4f}) ---")
            print(
                f"Guideline Name: {original_metadata.get('guideline_title', 'N/A')}")
            # ADDED: Display the URL for the guideline
            print(
                f"Guideline URL: {original_metadata.get('source_page_url', 'N/A')}")
            # print(f"Source File: {original_metadata.get('filename', 'N/A')}")
            print("\nRetrieved Context:")
            print(doc.page_content)
            print("-" * 30)


if __name__ == "__main__":
    main()
