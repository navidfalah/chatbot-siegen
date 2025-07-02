# ==============================================================================
# SCRIPT: rag_retriever.py
#
# PURPOSE:
#   - Provides a unified retrieval function for RAG applications
#   - Combines data from both AWMF (medical guidelines) and MUT_ATLAS (locations)
#   - Automatically detects query intent to search the appropriate database
#   - Returns formatted results ready for RAG consumption
#
# HOW TO USE:
#   1. Import the retrieve_content function from this module
#   2. Call the function with your query and optional parameters
#   3. Process the returned results in your RAG application
#
# EXAMPLE:
#   from rag_retriever import retrieve_content
#
#   results = retrieve_content(
#       query="Wo finde ich psychologische Hilfe in Siegen?",
#       top_k=3,
#       qdrant_host="localhost",
#       qdrant_port=6333
#   )
# ==============================================================================
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient, models
from mxbai_rerank import MxbaiRerankV2

# --- Configuration ---
DENSE_EMBEDDING_MODEL_NAME = 'mixedbread-ai/deepset-mxbai-embed-de-large-v1'
SPARSE_EMBEDDING_MODEL_NAME = 'Qdrant/bm25'
RERANKER_MODEL_NAME = 'mixedbread-ai/mxbai-rerank-large-v2'

# Collection names
GUIDELINES_COLLECTION = "awmf-hybrid-rerank-de"
LOCATIONS_COLLECTION = "locations-v2"

# Vector names (shared across collections)
DENSE_VECTOR_NAME = "dense_vector"
SPARSE_VECTOR_NAME = "sparse_vector"

# Candidate count for initial retrieval before reranking
CANDIDATE_COUNT = 25

# Location-related keywords to identify location queries
LOCATION_KEYWORDS = [
    "wo", "standort", "adresse", "kontakt", "telefon", "email",
    "ort", "stadt", "einrichtung", "beratungsstelle", "hilfe",
    "therapeut", "arzt", "klinik", "praxis", "zentrum", "anlaufstelle"
]

# Health guidelines keywords to identify medical guideline queries
GUIDELINE_KEYWORDS = [
    "krankheit", "symptom", "therapie", "behandlung", "medikament",
    "diagnose", "leitlinie", "empfehlung", "patient", "erkrankung",
    "syndrom", "vorsorge", "heilung", "verlauf", "prognose"
]

# Common city names in Germany to help with location detection
COMMON_CITIES = [
    "berlin", "hamburg", "münchen", "köln", "frankfurt", "stuttgart",
    "düsseldorf", "dortmund", "essen", "leipzig", "bremen", "dresden",
    "hannover", "nürnberg", "duisburg", "bochum", "wuppertal", "bielefeld",
    "bonn", "münster", "mannheim", "karlsruhe", "augsburg", "wiesbaden",
    "mönchengladbach", "gelsenkirchen", "aachen", "braunschweig", "kiel",
    "chemnitz", "halle", "magdeburg", "freiburg", "krefeld", "mainz",
    "lübeck", "erfurt", "oberhausen", "rostock", "kassel", "hagen",
    "potsdam", "saarbrücken", "hamm", "ludwigshafen", "oldenburg",
    "leverkusen", "osnabrück", "solingen", "heidelberg", "herne",
    "neuss", "darmstadt", "paderborn", "regensburg", "ingolstadt",
    "würzburg", "wolfsburg", "ulm", "bottrop", "pforzheim", "recklinghausen",
    "göttingen", "erlangen", "trier", "salzgitter", "siegen", "koblenz"
]

# Common health topics in German to help with medical guideline detection
HEALTH_TOPICS = [
    "diabetes", "depression", "angst", "krebs", "onkologie", "herzinfarkt",
    "schlaganfall", "bluthochdruck", "hypertonie", "asthma", "allergie",
    "rheuma", "arthritis", "osteoporose", "demenz", "alzheimer", "parkinson",
    "migräne", "kopfschmerz", "rückenschmerz", "schmerztherapie", "sucht",
    "adipositas", "übergewicht", "ernährung", "impfung", "grippe", "schwangerschaft",
    "geburt", "stillzeit", "kinderkrankheit", "neurodermitis", "psoriasis",
    "hauterkrankung", "schlafstörung", "burnout", "stress", "trauma", "psychotherapie",
    "rehabilitation", "pflege", "behinderung", "multiple sklerose", "epilepsie",
    "autismus", "adhd", "adhs", "schilddrüse", "lunge", "copd", "leber", "niere",
    "dialyse", "transplantation", "immunsystem", "hiv", "aids", "infektion"
]


def detect_locations_in_query(query: str) -> List[str]:
    """
    Detects potential city names in the query by looking for matches with common German cities.
    This approach is much simpler and more reliable than complex regex patterns.

    Args:
        query: The user query

    Returns:
        A list of detected city names (may be empty if none detected)
    """
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()

    # Find all city names mentioned in the query
    found_cities = []

    # First try to find cities with word boundaries to avoid partial matches
    for city in COMMON_CITIES:
        # Look for the city with word boundaries (space, punctuation, or string start/end)
        if re.search(r'(?:^|\W)' + re.escape(city) + r'(?:$|\W)', query_lower):
            found_cities.append(city)

    # If no cities found with strict matching, try more flexible matching as fallback
    if not found_cities:
        # Try matching with city names that might be part of compound words
        for city in COMMON_CITIES:
            # Only consider cities with >3 chars to avoid false positives
            if city in query_lower and len(city) > 3:
                found_cities.append(city)

    return found_cities


def detect_query_intent(query: str, detected_locations: List[str], detected_topics: List[str]) -> str:
    """
    Detects whether the query is likely about locations or medical guidelines.

    Args:
        query: The user query
        detected_locations: List of locations detected in the query
        detected_topics: List of health topics detected in the query

    Returns:
        String indicating detected intent: "location", "guideline", or "both"
    """
    query_lower = query.lower()

    # Check for explicit location markers
    location_score = sum(
        1 for keyword in LOCATION_KEYWORDS if keyword in query_lower)

    # Check for explicit guideline markers
    guideline_score = sum(
        1 for keyword in GUIDELINE_KEYWORDS if keyword in query_lower)

    # Add weight for detected cities
    if detected_locations:
        location_score += 2

    # Add weight for detected health topics
    if detected_topics:
        guideline_score += 2

    # Also look for location-related prepositions with potential cities
    prepositions = ["in", "bei", "aus", "für", "von", "nahe"]
    for prep in prepositions:
        if f" {prep} " in f" {query_lower} ":
            location_score += 0.5

    # Decision logic
    if location_score > 0 and guideline_score == 0:
        return "location"
    elif guideline_score > 0 and location_score == 0:
        return "guideline"
    elif location_score > guideline_score:
        return "location"
    elif guideline_score > location_score:
        return "guideline"
    else:
        return "both"  # Default to searching both if unclear


def initialize_retrieval_components(
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333
) -> Tuple[HuggingFaceEmbeddings, FastEmbedSparse, QdrantClient, MxbaiRerankV2]:
    """
    Initializes all components needed for retrieval.

    Args:
        qdrant_host: Host address for Qdrant database
        qdrant_port: Port for Qdrant database

    Returns:
        Tuple containing initialized components:
        (dense_embeddings, sparse_embeddings, qdrant_client, reranker)
    """
    # Initialize dense embeddings model
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=DENSE_EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Initialize sparse embeddings model
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL_NAME)

    # Connect to Qdrant
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Initialize reranker
    reranker = MxbaiRerankV2(RERANKER_MODEL_NAME)

    return dense_embeddings, sparse_embeddings, qdrant_client, reranker


def retrieve_from_collection(
    query: str,
    collection_name: str,
    dense_embeddings: HuggingFaceEmbeddings,
    sparse_embeddings: FastEmbedSparse,
    qdrant_client: QdrantClient,
    reranker: MxbaiRerankV2,
    top_k: int = 5,
    batch_size: int = 4,
    filter_condition: Optional[models.Filter] = None,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Retrieves documents from a specific collection with hybrid search and reranking.

    Args:
        query: User query
        collection_name: Name of the Qdrant collection to search
        dense_embeddings: Initialized dense embeddings model
        sparse_embeddings: Initialized sparse embeddings model
        qdrant_client: Initialized Qdrant client
        reranker: Initialized reranker model
        top_k: Number of results to return after reranking
        batch_size: Batch size for reranker processing
        filter_condition: Optional filter to apply to the search
        debug: Whether to print debug information

    Returns:
        List of dictionaries containing the retrieved documents and metadata
    """
    # Create vector store
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )

    # Retrieve candidate documents with scores
    if debug:
        print(
            f"[DEBUG] Searching collection '{collection_name}' with query: '{query}'")
        if filter_condition:
            print(f"[DEBUG] Using filter: {filter_condition}")

    candidate_results = vector_store.similarity_search_with_score(
        query,
        k=CANDIDATE_COUNT,
        filter=filter_condition
    )

    if not candidate_results:
        if debug:
            print(
                f"[DEBUG] No results found from initial search in '{collection_name}'")
        return []

    if debug:
        print(
            f"[DEBUG] Found {len(candidate_results)} initial candidates in '{collection_name}'")

    # Separate documents and their initial scores
    candidate_docs = [res[0] for res in candidate_results]
    initial_scores = {doc.page_content: res[1] for doc, res in zip(
        candidate_docs, candidate_results)}

    # Rerank candidates
    if debug:
        print(
            f"[DEBUG] Reranking {len(candidate_docs)} candidates from '{collection_name}'")

    doc_texts = [doc.page_content for doc in candidate_docs]
    reranked_results = reranker.rank(
        query=query,
        documents=doc_texts,
        return_documents=False,
        top_k=top_k,
        batch_size=batch_size
    )

    if debug:
        print(
            f"[DEBUG] Got {len(reranked_results)} reranked results from '{collection_name}'")

    # Combine original docs with new rerank scores
    final_results = []

    for result in reranked_results:
        original_doc = candidate_docs[result.index]
        document_content = original_doc.page_content
        metadata = original_doc.metadata.copy()

        # Add scoring information
        metadata['rerank_score'] = result.score
        metadata['initial_hybrid_score'] = initial_scores.get(
            document_content, 'N/A')

        # Format results based on collection type
        if collection_name == GUIDELINES_COLLECTION:
            final_results.append({
                'content': document_content,
                'type': 'guideline',
                'title': metadata.get('guideline_title', 'Unknown Guideline'),
                'url': metadata.get('source_page_url', ''),
                'metadata': metadata
            })
        elif collection_name == LOCATIONS_COLLECTION:
            address = metadata.get('address', {})
            contact = metadata.get('contact', {})

            address_str = f"{address.get('street', '')} {address.get('house_number', '')}, " \
                f"{address.get('zip_code', '')} {address.get('city', '')}"

            final_results.append({
                'content': document_content,
                'type': 'location',
                'name': metadata.get('name', 'Unknown Location'),
                'address': address_str.strip(),
                'city': address.get('city', ''),
                'email': contact.get('email', ''),
                'phone': contact.get('phone', ''),
                'website': contact.get('homepage', ''),
                'metadata': metadata
            })

    return final_results


def retrieve_content(
    query: str,
    top_k: int = 5,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    batch_size: int = 4,
    force_collection: Optional[str] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Main retrieval function that automatically determines which collection to query
    based on the query content and returns relevant information.

    Args:
        query: User query string
        top_k: Number of results to return from each collection
        qdrant_host: Qdrant host address
        qdrant_port: Qdrant port
        batch_size: Batch size for reranker
        force_collection: Force using a specific collection ("guideline", "location", or "both")
        debug: Whether to print debug information

    Returns:
        Dictionary containing:
            - query_info: Information about query processing
            - results: Combined results from relevant collections
    """
    # Initialize retrieval components
    dense_embeddings, sparse_embeddings, qdrant_client, reranker = initialize_retrieval_components(
        qdrant_host, qdrant_port
    )

    # Detect locations and health topics in the query
    detected_locations = detect_locations_in_query(query)
    detected_topics = detect_health_topics_in_query(query)

    # Determine query intent using detected locations and topics
    query_intent = force_collection if force_collection else detect_query_intent(
        query, detected_locations, detected_topics)

    if debug:
        print(f"\n[DEBUG] Original query: '{query}'")
        print(f"[DEBUG] Detected locations: {detected_locations}")
        print(f"[DEBUG] Detected health topics: {detected_topics}")
        print(f"[DEBUG] Query intent: {query_intent}")

    # Initialize results container
    results = {
        'query_info': {
            'original_query': query,
            'detected_locations': detected_locations,
            'detected_topics': detected_topics,
            'search_type': query_intent
        },
        'results': []
    }

    # SIMPLIFIED APPROACH: Don't use explicit filtering which can be too restrictive
    # Instead, rely on the semantic search to find relevant results, since location
    # names will be part of the query and should naturally rank relevant results higher

    # Query appropriate collection(s) based on intent
    if query_intent in ["location", "both"]:
        if debug:
            print(
                f"[DEBUG] Searching LOCATIONS collection: {LOCATIONS_COLLECTION}")

        # Search without explicit filters to allow for more flexible matching
        location_results = retrieve_from_collection(
            query,  # Use the original query for semantic search
            LOCATIONS_COLLECTION,
            dense_embeddings,
            sparse_embeddings,
            qdrant_client,
            reranker,
            top_k=top_k * 2,  # Retrieve more results for better coverage
            batch_size=batch_size,
            filter_condition=None,  # No explicit filtering
            debug=debug
        )

        # If locations were detected, post-process results to prioritize those locations
        if detected_locations and location_results:
            if debug:
                print(
                    f"[DEBUG] Post-processing {len(location_results)} location results to prioritize detected locations")

            # Boost scores for results that match detected locations
            for result in location_results:
                # Extract city from the result
                city = result.get('city', '').lower()
                name = result.get('name', '').lower()
                address = result.get('address', '').lower()

                # Check if any detected location appears in the result
                for location in detected_locations:
                    location_match_score = 0

                    # City exact match is highest priority
                    if location == city:
                        location_match_score = 1.0
                    # Location in name
                    elif location in name:
                        location_match_score = 0.8
                    # Location in address
                    elif location in address:
                        location_match_score = 0.7

                    if location_match_score > 0:
                        # Boost the rerank score by multiplying with the location match score
                        original_score = result['metadata']['rerank_score']
                        result['metadata']['rerank_score'] = original_score * \
                            (1 + location_match_score)
                        result['metadata']['location_boost'] = location_match_score

                        if debug:
                            print(f"[DEBUG] Boosted score for result matching '{location}'. " +
                                  f"Original: {original_score:.4f}, Boosted: {result['metadata']['rerank_score']:.4f}")

        if debug:
            print(f"[DEBUG] Found {len(location_results)} location results")

        results['results'].extend(location_results)

    if query_intent in ["guideline", "both"]:
        if debug:
            print(
                f"[DEBUG] Searching GUIDELINES collection: {GUIDELINES_COLLECTION}")

        guideline_results = retrieve_from_collection(
            query,  # Use the original query for semantic search
            GUIDELINES_COLLECTION,
            dense_embeddings,
            sparse_embeddings,
            qdrant_client,
            reranker,
            top_k=top_k * 2,  # Retrieve more results for better coverage
            batch_size=batch_size,
            debug=debug
        )

        # If health topics were detected, post-process results to prioritize those topics
        if detected_topics and guideline_results:
            if debug:
                print(
                    f"[DEBUG] Post-processing {len(guideline_results)} guideline results to prioritize detected health topics")

            # Boost scores for results that match detected health topics
            for result in guideline_results:
                # Extract title and content from the result
                title = result.get('title', '').lower()
                content = result.get('content', '').lower()

                # Check if any detected topic appears in the result
                for topic in detected_topics:
                    topic_match_score = 0

                    # Topic in title is highest priority
                    if topic in title:
                        topic_match_score = 1.0
                    # Topic in content
                    elif topic in content:
                        # Count occurrences to give higher boost to documents that mention the topic more
                        count = content.count(topic)
                        # Scale by log to avoid excessive boosting for repetition
                        topic_match_score = 0.5 * min(1 + (count / 10), 1.5)

                    if topic_match_score > 0:
                        # Boost the rerank score by multiplying with the topic match score
                        original_score = result['metadata']['rerank_score']
                        result['metadata']['rerank_score'] = original_score * \
                            (1 + topic_match_score)
                        result['metadata']['topic_boost'] = topic_match_score

                        if debug:
                            print(f"[DEBUG] Boosted score for result matching '{topic}'. " +
                                  f"Original: {original_score:.4f}, Boosted: {result['metadata']['rerank_score']:.4f}")

        if debug:
            print(f"[DEBUG] Found {len(guideline_results)} guideline results")

        results['results'].extend(guideline_results)

    # Sort combined results by rerank score
    if results['results']:
        results['results'] = sorted(
            results['results'],
            key=lambda x: x['metadata']['rerank_score'],
            reverse=True
        )[:top_k]

    return results


def detect_health_topics_in_query(query: str) -> List[str]:
    """
    Detects potential health topics in the query by looking for matches with common health conditions.

    Args:
        query: The user query

    Returns:
        A list of detected health topics (may be empty if none detected)
    """
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()

    # Find all health topics mentioned in the query
    found_topics = []

    # First try to find topics with word boundaries for more precise matching
    for topic in HEALTH_TOPICS:
        # Look for the topic with word boundaries (space, punctuation, or string start/end)
        if re.search(r'(?:^|\W)' + re.escape(topic) + r'(?:$|\W)', query_lower):
            found_topics.append(topic)

    # If no topics found with strict matching, try more flexible matching as fallback
    if not found_topics:
        # Try matching with topics that might be part of compound words
        for topic in HEALTH_TOPICS:
            # Only consider topics with >4 chars to avoid false positives
            if topic in query_lower and len(topic) > 4:
                found_topics.append(topic)

    return found_topics
