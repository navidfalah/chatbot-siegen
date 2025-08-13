from mxbai_rerank import MxbaiRerankV2
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any, Tuple
import torch
import re
import os
import gc
import logging
import time
import psutil
from datetime import datetime
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# Configure logging based on environment variable
VERBOSE_LOGGING = os.environ.get("VERBOSE_LOGGING", "").lower() in [
    "true", "1", "yes", "y"]

logging.basicConfig(
    level=logging.DEBUG if VERBOSE_LOGGING else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def log_gpu_memory(prefix=""):
    """Log GPU memory usage if available."""
    if not VERBOSE_LOGGING:
        return

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logging.debug(
            f"{prefix} GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max Allocated={max_allocated:.2f}GB")
        for i in range(torch.cuda.device_count()):
            logging.debug(
                f"GPU {i} - Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f}GB")
    else:
        logging.debug(f"{prefix} No GPU available")


def log_system_memory():
    """Log system memory usage."""
    if not VERBOSE_LOGGING:
        return

    mem = psutil.virtual_memory()
    logging.debug(
        f"System Memory: Total={mem.total/(1024**3):.2f}GB, Available={mem.available/(1024**3):.2f}GB, Used={mem.used/(1024**3):.2f}GB ({mem.percent}%)")


# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
DENSE_EMBEDDING_MODEL_NAME = 'mixedbread-ai/deepset-mxbai-embed-de-large-v1'
SPARSE_EMBEDDING_MODEL_NAME = 'Qdrant/bm25'
RERANKER_MODEL_NAME = 'mixedbread-ai/mxbai-rerank-base-v2'
GUIDELINES_COLLECTION = "awmf-hybrid-rerank-de"
LOCATIONS_COLLECTION = "locations-v2"
DENSE_VECTOR_NAME = "dense_vector"
SPARSE_VECTOR_NAME = "sparse_vector"
CANDIDATE_COUNT = 25

LOCATION_KEYWORDS = ["wo", "standort", "adresse", "kontakt", "telefon", "email", "ort", "stadt", "einrichtung",
                     "beratungsstelle", "hilfe", "therapeut", "arzt", "klinik", "praxis", "zentrum", "anlaufstelle"]
GUIDELINE_KEYWORDS = ["krankheit", "symptom", "therapie", "behandlung", "medikament", "diagnose",
                      "leitlinie", "empfehlung", "patient", "erkrankung", "syndrom", "vorsorge", "heilung", "verlauf", "prognose"]
COMMON_CITIES = ["berlin", "hamburg", "münchen", "köln", "frankfurt", "stuttgart", "düsseldorf", "dortmund", "essen", "leipzig", "bremen", "dresden", "hannover", "nürnberg", "duisburg", "bochum", "wuppertal", "bielefeld", "bonn", "münster", "mannheim", "karlsruhe", "augsburg", "wiesbaden", "mönchengladbach", "gelsenkirchen", "aachen", "braunschweig", "kiel", "chemnitz", "halle", "magdeburg", "freiburg",
                 "krefeld", "mainz", "lübeck", "erfurt", "oberhausen", "rostock", "kassel", "hagen", "potsdam", "saarbrücken", "hamm", "ludwigshafen", "oldenburg", "leverkusen", "osnabrück", "solingen", "heidelberg", "herne", "neuss", "darmstadt", "paderborn", "regensburg", "ingolstadt", "würzburg", "wolfsburg", "ulm", "bottrop", "pforzheim", "recklinghausen", "göttingen", "erlangen", "trier", "salzgitter", "siegen", "koblenz"]
HEALTH_TOPICS = ["diabetes", "depression", "angst", "krebs", "onkologie", "herzinfarkt", "schlaganfall", "bluthochdruck", "hypertonie", "asthma", "allergie", "rheuma", "arthritis", "osteoporose", "demenz", "alzheimer", "parkinson", "migräne", "kopfschmerz", "rückenschmerz", "schmerztherapie", "sucht", "adipositas", "übergewicht", "ernährung", "impfung", "grippe", "schwangerschaft",
                 "geburt", "stillzeit", "kinderkrankheit", "neurodermitis", "psoriasis", "hauterkrankung", "schlafstörung", "burnout", "stress", "trauma", "psychotherapie", "rehabilitation", "pflege", "behinderung", "multiple sklerose", "epilepsie", "autismus", "adhd", "adhs", "schilddrüse", "lunge", "copd", "leber", "niere", "dialyse", "transplantation", "immunsystem", "hiv", "aids", "infektion"]


def retriever_worker(request_queue, response_queue):
    logging.info("Retriever worker process started. Initializing models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Retriever worker using device: {device}")

    # Log initial system state (only in verbose mode)
    log_system_memory()
    log_gpu_memory("Initial")

    start_time = time.time()
    try:
        if VERBOSE_LOGGING:
            logging.debug("Starting initialization of retrieval components...")

        dense_embeddings, sparse_embeddings, qdrant_client, reranker = initialize_retrieval_components(
            device)

        logging.info(
            f"Models and Qdrant client initialized successfully in {time.time() - start_time:.2f} seconds")

        # Log memory usage after initialization (only in verbose mode)
        log_system_memory()
        log_gpu_memory("Post-initialization")

        # Log CUDA device information if available (basic info always, details only in verbose mode)
        if device == "cuda":
            if VERBOSE_LOGGING:
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    logging.debug(
                        f"GPU {i}: {device_props.name}, Total Memory: {device_props.total_memory / (1024**3):.2f}GB")
            else:
                logging.info(
                    f"Using GPU with {torch.cuda.device_count()} device(s)")

    except Exception as e:
        logging.error(f"Worker initialization failed: {str(e)}")
        if VERBOSE_LOGGING:
            logging.error("Full error traceback:", exc_info=True)
        response_queue.put({"error": "InitializationError",
                           "message": f"Worker initialization failed: {e}"})
        return

    while True:
        request_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        try:
            if VERBOSE_LOGGING:
                logging.debug(f"[{request_id}] Waiting for next request...")
            request = request_queue.get()

            if request is None:
                logging.info("Received stop signal. Shutting down worker.")
                break

            if request.get('type') == 'health_check':
                if VERBOSE_LOGGING:
                    logging.debug(
                        f"[{request_id}] Processing health check request")
                    log_gpu_memory("Health check")
                response_queue.put({
                    "status": "active",
                    "device": device,
                    "device_name": torch.cuda.get_device_name(0) if device == "cuda" else None
                })
                continue

            # Log request details (only in verbose mode)
            log_system_memory()
            log_gpu_memory(f"[{request_id}] Before processing")

            query = request.get('query', '')
            top_k = request.get('top_k', 5)
            force_collection = request.get('collection', None)

            # Log request details (basic in normal mode, detailed in verbose mode)
            if VERBOSE_LOGGING:
                logging.debug(
                    f"[{request_id}] Processing retrieval request: query='{query[:50]}{'...' if len(query) > 50 else ''}', top_k={top_k}, collection={force_collection}")
            else:
                logging.info(
                    f"Processing query (length: {len(query)}, top_k: {top_k})")

            start_time = time.time()
            results = retrieve_content(
                dense_embeddings=dense_embeddings,
                sparse_embeddings=sparse_embeddings,
                qdrant_client=qdrant_client,
                reranker=reranker,
                request_id=request_id,  # Pass request_id for logging
                **request
            )
            processing_time = time.time() - start_time

            # Always log completion, but with different detail levels
            if VERBOSE_LOGGING:
                logging.debug(
                    f"[{request_id}] Request processed in {processing_time:.2f} seconds, found {len(results.get('results', []))} results")
                log_gpu_memory(f"[{request_id}] After processing")
            else:
                logging.info(
                    f"Request completed in {processing_time:.2f}s, found {len(results.get('results', []))} results")

            response_queue.put(results)

        except torch.cuda.OutOfMemoryError as e:
            # For OOM errors, always log detailed information regardless of verbose mode
            logging.error(f"[{request_id}] GPU OUT OF MEMORY ERROR: {str(e)}")
            if VERBOSE_LOGGING:
                logging.error("Full error traceback:", exc_info=True)

            # Always log memory status for OOM errors
            mem = psutil.virtual_memory()
            logging.error(
                f"[{request_id}] OOM ERROR - System memory: {mem.available/(1024**3):.2f}GB available ({mem.percent}% used)")

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    free_mem = torch.cuda.memory_reserved(
                        i) - torch.cuda.memory_allocated(i)
                    total_mem = torch.cuda.get_device_properties(
                        i).total_memory
                    logging.error(
                        f"[{request_id}] OOM ERROR - GPU {i} memory: {free_mem/(1024**3):.2f}GB free of {total_mem/(1024**3):.2f}GB total")

            log_gpu_memory(f"[{request_id}] After OOM error")
            error_response = {"error": "OutOfMemoryError",
                              "message": f"GPU memory exhausted. The worker may need to be restarted. Details: {str(e)}"}
            response_queue.put(error_response)
        except RuntimeError as e:
            if "CUDA" in str(e):
                logging.error(f"[{request_id}] CUDA RUNTIME ERROR: {str(e)}")
                if VERBOSE_LOGGING:
                    logging.error("Full error traceback:", exc_info=True)
                log_gpu_memory(f"[{request_id}] After CUDA error")
                error_response = {
                    "error": "CUDAError", "message": f"A CUDA-related runtime error occurred: {e}"}
            else:
                logging.error(f"[{request_id}] RUNTIME ERROR: {str(e)}")
                if VERBOSE_LOGGING:
                    logging.error("Full error traceback:", exc_info=True)
                error_response = {
                    "error": "GenericRuntimeError", "message": str(e)}
            response_queue.put(error_response)
        except Exception as e:
            logging.error(f"[{request_id}] UNEXPECTED ERROR: {str(e)}")
            if VERBOSE_LOGGING:
                logging.error("Full error traceback:", exc_info=True)
            error_response = {"error": "UnknownError", "message": str(e)}
            response_queue.put(error_response)

        finally:
            if device == 'cuda':
                if VERBOSE_LOGGING:
                    logging.debug(f"[{request_id}] Clearing CUDA cache")
                torch.cuda.empty_cache()
                # Additional cleanup to help prevent memory leaks
                gc.collect()


def initialize_retrieval_components(device: str):
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=DENSE_EMBEDDING_MODEL_NAME, model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings': True}
    )
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL_NAME)

    # Read Qdrant configuration from environment variables
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    reranker = MxbaiRerankV2(RERANKER_MODEL_NAME, device=device)
    return dense_embeddings, sparse_embeddings, qdrant_client, reranker


def retrieve_from_collection(query: str, collection_name: str, dense_embeddings: HuggingFaceEmbeddings, sparse_embeddings: FastEmbedSparse, qdrant_client: QdrantClient, reranker: MxbaiRerankV2, top_k: int, batch_size: int, filter_condition: Optional[models.Filter] = None, debug: bool = False, request_id: str = "") -> List[Dict[str, Any]]:
    """Retrieves documents from a collection with detailed memory tracking."""
    step_times = {}
    start_time = time.time()

    logging.info(f"[{request_id}] Creating vector store for {collection_name}")
    vector_store = QdrantVectorStore(
        client=qdrant_client, collection_name=collection_name, embedding=dense_embeddings, sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID, vector_name=DENSE_VECTOR_NAME, sparse_vector_name=SPARSE_VECTOR_NAME,
    )
    step_times['vector_store_creation'] = time.time() - start_time

    # Track memory after vector store creation
    log_gpu_memory(
        f"[{request_id}] After vector store creation for {collection_name}")

    # Initial search
    search_start = time.time()
    logging.info(
        f"[{request_id}] Performing similarity search on {collection_name}")
    try:
        candidate_results = vector_store.similarity_search_with_score(
            query, k=CANDIDATE_COUNT, filter=filter_condition)
        step_times['similarity_search'] = time.time() - search_start
        logging.info(
            f"[{request_id}] Found {len(candidate_results)} candidates in {step_times['similarity_search']:.2f}s")
    except Exception as e:
        logging.error(
            f"[{request_id}] Error in similarity search: {str(e)}", exc_info=True)
        log_gpu_memory(f"[{request_id}] After similarity search error")
        raise

    if not candidate_results:
        logging.info(f"[{request_id}] No results found in {collection_name}")
        return []

    # Process candidates
    process_start = time.time()
    candidate_docs = [res[0] for res in candidate_results]
    initial_scores = {doc.page_content: res[1] for doc, res in zip(
        candidate_docs, candidate_results)}
    doc_texts = [doc.page_content for doc in candidate_docs]
    step_times['process_candidates'] = time.time() - process_start

    # Track memory before reranking (most memory-intensive operation)
    log_gpu_memory(f"[{request_id}] Before reranking for {collection_name}")

    # Reranking
    rerank_start = time.time()
    logging.info(
        f"[{request_id}] Reranking {len(doc_texts)} candidates with batch size {batch_size}")
    try:
        reranked_results = reranker.rank(
            query=query, documents=doc_texts, return_documents=False, top_k=top_k, batch_size=batch_size)
        step_times['reranking'] = time.time() - rerank_start
        logging.info(
            f"[{request_id}] Reranking completed in {step_times['reranking']:.2f}s")
    except torch.cuda.OutOfMemoryError as e:
        logging.error(
            f"[{request_id}] OUT OF MEMORY during reranking: {str(e)}", exc_info=True)
        log_gpu_memory(f"[{request_id}] After reranking OOM error")
        # Try to reduce batch size and retry once
        if batch_size > 1:
            reduced_batch = max(1, batch_size // 2)
            logging.info(
                f"[{request_id}] Retrying with reduced batch size: {reduced_batch}")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                reranked_results = reranker.rank(
                    query=query, documents=doc_texts, return_documents=False, top_k=top_k, batch_size=reduced_batch)
                logging.info(
                    f"[{request_id}] Reranking with reduced batch size succeeded")
            except Exception as retry_e:
                logging.error(
                    f"[{request_id}] Retry failed: {str(retry_e)}", exc_info=True)
                raise
        else:
            raise
    except Exception as e:
        logging.error(
            f"[{request_id}] Error during reranking: {str(e)}", exc_info=True)
        log_gpu_memory(f"[{request_id}] After reranking error")
        raise

    # Track memory after reranking
    log_gpu_memory(f"[{request_id}] After reranking for {collection_name}")

    # Process final results
    format_start = time.time()
    final_results = []
    for result in reranked_results:
        original_doc = candidate_docs[result.index]
        document_content = original_doc.page_content
        metadata = original_doc.metadata.copy()
        metadata['rerank_score'] = result.score
        metadata['initial_hybrid_score'] = initial_scores.get(
            document_content, 'N/A')

        if collection_name == GUIDELINES_COLLECTION:
            final_results.append({'content': document_content, 'type': 'guideline', 'title': metadata.get(
                'guideline_title', 'Unknown Guideline'), 'url': metadata.get('source_page_url', ''), 'metadata': metadata})
        elif collection_name == LOCATIONS_COLLECTION:
            address = metadata.get('address', {})
            contact = metadata.get('contact', {})
            address_str = f"{address.get('street', '')} {address.get('house_number', '')}, {address.get('zip_code', '')} {address.get('city', '')}"
            final_results.append({'content': document_content, 'type': 'location', 'name': metadata.get('name', 'Unknown Location'), 'address': address_str.strip(
            ), 'city': address.get('city', ''), 'email': contact.get('email', ''), 'phone': contact.get('phone', ''), 'website': contact.get('homepage', ''), 'metadata': metadata})
    step_times['format_results'] = time.time() - format_start

    total_time = time.time() - start_time
    logging.info(f"[{request_id}] {collection_name} retrieval completed in {total_time:.2f}s: " +
                 f"Vector store: {step_times.get('vector_store_creation', 0):.2f}s, " +
                 f"Search: {step_times.get('similarity_search', 0):.2f}s, " +
                 f"Reranking: {step_times.get('reranking', 0):.2f}s, " +
                 f"Formatting: {step_times.get('format_results', 0):.2f}s")

    # Free memory
    del vector_store
    del candidate_results
    del candidate_docs
    del doc_texts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_results


def retrieve_content(query: str, top_k: int, dense_embeddings: HuggingFaceEmbeddings, sparse_embeddings: FastEmbedSparse, qdrant_client: QdrantClient, reranker: MxbaiRerankV2, force_collection: Optional[str] = None, debug: bool = False, batch_size: int = 4, **kwargs) -> Dict[str, Any]:
    request_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_gpu_memory(
            f"[{request_id}] After initial cleanup in retrieve_content")

    logging.info(
        f"[{request_id}] Starting content retrieval for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")

    # Track time for each step
    step_start = time.time()
    detected_locations = detect_locations_in_query(query)
    detected_topics = detect_health_topics_in_query(query)
    query_intent = force_collection if force_collection else detect_query_intent(
        query, detected_locations, detected_topics)

    logging.info(f"[{request_id}] Query analysis completed in {time.time() - step_start:.2f}s: intent={query_intent}, locations={detected_locations}, topics={detected_topics}")

    results = {'query_info': {'original_query': query, 'detected_locations': detected_locations,
                              'detected_topics': detected_topics, 'search_type': query_intent}, 'results': []}

    if query_intent in ["location", "both"]:
        step_start = time.time()
        logging.info(f"[{request_id}] Retrieving from LOCATIONS_COLLECTION")
        log_gpu_memory(f"[{request_id}] Before location retrieval")

        location_results = retrieve_from_collection(query, LOCATIONS_COLLECTION, dense_embeddings, sparse_embeddings,
                                                    qdrant_client, reranker, top_k=top_k * 2, batch_size=batch_size,
                                                    filter_condition=None, debug=debug, request_id=request_id)

        logging.info(
            f"[{request_id}] Location retrieval completed in {time.time() - step_start:.2f}s, found {len(location_results)} results")
        log_gpu_memory(f"[{request_id}] After location retrieval")

        if detected_locations and location_results:
            boost_start = time.time()
            for result in location_results:
                for location in detected_locations:
                    location_match_score = 0
                    if location == result.get('city', '').lower():
                        location_match_score = 1.0
                    elif location in result.get('name', '').lower():
                        location_match_score = 0.8
                    elif location in result.get('address', '').lower():
                        location_match_score = 0.7
                    if location_match_score > 0:
                        result['metadata']['rerank_score'] *= (
                            1 + location_match_score)
                        result['metadata']['location_boost'] = location_match_score
                        break
            logging.info(
                f"[{request_id}] Location boosting completed in {time.time() - boost_start:.2f}s")

        results['results'].extend(location_results)

        # Force garbage collection after adding results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if query_intent in ["guideline", "both"]:
        step_start = time.time()
        logging.info(f"[{request_id}] Retrieving from GUIDELINES_COLLECTION")
        log_gpu_memory(f"[{request_id}] Before guideline retrieval")

        guideline_results = retrieve_from_collection(query, GUIDELINES_COLLECTION, dense_embeddings,
                                                     sparse_embeddings, qdrant_client, reranker,
                                                     top_k=top_k * 2, batch_size=batch_size,
                                                     debug=debug, request_id=request_id)

        logging.info(
            f"[{request_id}] Guideline retrieval completed in {time.time() - step_start:.2f}s, found {len(guideline_results)} results")
        log_gpu_memory(f"[{request_id}] After guideline retrieval")

        if detected_topics and guideline_results:
            boost_start = time.time()
            for result in guideline_results:
                for topic in detected_topics:
                    topic_match_score = 0
                    if topic in result.get('title', '').lower():
                        topic_match_score = 1.0
                    elif topic in result.get('content', '').lower():
                        topic_match_score = 0.5 * \
                            min(1 + (result.get('content',
                                '').lower().count(topic) / 10), 1.5)
                    if topic_match_score > 0:
                        result['metadata']['rerank_score'] *= (
                            1 + topic_match_score)
                        result['metadata']['topic_boost'] = topic_match_score
                        break
            logging.info(
                f"[{request_id}] Topic boosting completed in {time.time() - boost_start:.2f}s")

        results['results'].extend(guideline_results)

        # Force garbage collection after adding results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if results['results']:
        sort_start = time.time()
        results['results'] = sorted(
            results['results'], key=lambda x: x['metadata']['rerank_score'], reverse=True)[:top_k]
        logging.info(
            f"[{request_id}] Results sorted in {time.time() - sort_start:.2f}s, returning {len(results['results'])} results")

    log_gpu_memory(f"[{request_id}] Before final cleanup")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    log_gpu_memory(f"[{request_id}] After final cleanup")

    return results


def detect_locations_in_query(query: str) -> List[str]:
    query_lower = query.lower()
    found_cities = []
    for city in COMMON_CITIES:
        if re.search(r'(?:^|\W)' + re.escape(city) + r'(?:$|\W)', query_lower):
            found_cities.append(city)
    if not found_cities:
        for city in COMMON_CITIES:
            if city in query_lower and len(city) > 3:
                found_cities.append(city)
    return found_cities


def detect_health_topics_in_query(query: str) -> List[str]:
    query_lower = query.lower()
    found_topics = []
    for topic in HEALTH_TOPICS:
        if re.search(r'(?:^|\W)' + re.escape(topic) + r'(?:$|\W)', query_lower):
            found_topics.append(topic)
    if not found_topics:
        for topic in HEALTH_TOPICS:
            if topic in query_lower and len(topic) > 4:
                found_topics.append(topic)
    return found_topics


def detect_query_intent(query: str, detected_locations: List[str], detected_topics: List[str]) -> str:
    query_lower = query.lower()
    location_score = sum(
        1 for keyword in LOCATION_KEYWORDS if keyword in query_lower)
    guideline_score = sum(
        1 for keyword in GUIDELINE_KEYWORDS if keyword in query_lower)
    if detected_locations:
        location_score += 2
    if detected_topics:
        guideline_score += 2
    for prep in ["in", "bei", "aus", "für", "von", "nahe"]:
        if f" {prep} " in f" {query_lower} ":
            location_score += 0.5
    if location_score > 0 and guideline_score == 0:
        return "location"
    if guideline_score > 0 and location_score == 0:
        return "guideline"
    if location_score > guideline_score:
        return "location"
    if guideline_score > location_score:
        return "guideline"
    return "both"
