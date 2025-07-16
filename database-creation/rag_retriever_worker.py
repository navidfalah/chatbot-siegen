import re
import os
import torch
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient, models
from mxbai_rerank import MxbaiRerankV2

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
    print("✅ Retriever worker process started. Initializing models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Retriever worker using device: {device}")

    try:
        dense_embeddings, sparse_embeddings, qdrant_client, reranker = initialize_retrieval_components(
            device)
        print("✅ Models and Qdrant client initialized successfully.")
    except Exception as e:
        print(f"❌ Worker initialization failed: {e}")
        response_queue.put({"error": "InitializationError",
                           "message": f"Worker initialization failed: {e}"})
        return

    while True:
        try:
            request = request_queue.get()
            if request is None:
                print("🛑 Received stop signal. Shutting down worker.")
                break

            if request.get('type') == 'health_check':
                response_queue.put({
                    "status": "active",
                    "device": device,
                    "device_name": torch.cuda.get_device_name(0) if device == "cuda" else None
                })
                continue

            results = retrieve_content(
                dense_embeddings=dense_embeddings,
                sparse_embeddings=sparse_embeddings,
                qdrant_client=qdrant_client,
                reranker=reranker,
                **request
            )
            response_queue.put(results)

        except torch.cuda.OutOfMemoryError:
            error_response = {"error": "OutOfMemoryError",
                              "message": "GPU memory exhausted. The worker may need to be restarted."}
            response_queue.put(error_response)
        except RuntimeError as e:
            if "CUDA" in str(e):
                error_response = {
                    "error": "CUDAError", "message": f"A CUDA-related runtime error occurred: {e}"}
            else:
                error_response = {
                    "error": "GenericRuntimeError", "message": str(e)}
            response_queue.put(error_response)
        except Exception as e:
            error_response = {"error": "UnknownError", "message": str(e)}
            response_queue.put(error_response)

        finally:
            if device == 'cuda':
                torch.cuda.empty_cache()


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


def retrieve_from_collection(query: str, collection_name: str, dense_embeddings: HuggingFaceEmbeddings, sparse_embeddings: FastEmbedSparse, qdrant_client: QdrantClient, reranker: MxbaiRerankV2, top_k: int, batch_size: int, filter_condition: Optional[models.Filter] = None, debug: bool = False) -> List[Dict[str, Any]]:
    vector_store = QdrantVectorStore(
        client=qdrant_client, collection_name=collection_name, embedding=dense_embeddings, sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID, vector_name=DENSE_VECTOR_NAME, sparse_vector_name=SPARSE_VECTOR_NAME,
    )
    candidate_results = vector_store.similarity_search_with_score(
        query, k=CANDIDATE_COUNT, filter=filter_condition)
    if not candidate_results:
        return []

    candidate_docs = [res[0] for res in candidate_results]
    initial_scores = {doc.page_content: res[1] for doc, res in zip(
        candidate_docs, candidate_results)}
    doc_texts = [doc.page_content for doc in candidate_docs]

    reranked_results = reranker.rank(
        query=query, documents=doc_texts, return_documents=False, top_k=top_k, batch_size=batch_size)
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
    return final_results


def retrieve_content(query: str, top_k: int, dense_embeddings: HuggingFaceEmbeddings, sparse_embeddings: FastEmbedSparse, qdrant_client: QdrantClient, reranker: MxbaiRerankV2, force_collection: Optional[str] = None, debug: bool = False, batch_size: int = 4, **kwargs) -> Dict[str, Any]:
    detected_locations = detect_locations_in_query(query)
    detected_topics = detect_health_topics_in_query(query)
    query_intent = force_collection if force_collection else detect_query_intent(
        query, detected_locations, detected_topics)

    results = {'query_info': {'original_query': query, 'detected_locations': detected_locations,
                              'detected_topics': detected_topics, 'search_type': query_intent}, 'results': []}

    if query_intent in ["location", "both"]:
        location_results = retrieve_from_collection(query, LOCATIONS_COLLECTION, dense_embeddings, sparse_embeddings,
                                                    qdrant_client, reranker, top_k=top_k * 2, batch_size=batch_size, filter_condition=None, debug=debug)
        if detected_locations and location_results:
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
        results['results'].extend(location_results)

    if query_intent in ["guideline", "both"]:
        guideline_results = retrieve_from_collection(query, GUIDELINES_COLLECTION, dense_embeddings,
                                                     sparse_embeddings, qdrant_client, reranker, top_k=top_k * 2, batch_size=batch_size, debug=debug)
        if detected_topics and guideline_results:
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
        results['results'].extend(guideline_results)

    if results['results']:
        results['results'] = sorted(
            results['results'], key=lambda x: x['metadata']['rerank_score'], reverse=True)[:top_k]
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
