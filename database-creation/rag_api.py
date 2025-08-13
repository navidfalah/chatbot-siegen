
import torch
from rag_retriever_worker import retriever_worker
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends, Header, status, Request
from fastapi.responses import JSONResponse
from typing import Dict, Optional, Any
from contextlib import contextmanager
import threading
import asyncio
import time
import logging
import traceback
import psutil
import gc
import platform
from datetime import datetime

import os
import secrets
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

# Add a message about verbose logging status
if VERBOSE_LOGGING:
    logging.info("Verbose logging is ENABLED")
else:
    logging.info(
        "Verbose logging is disabled. Set VERBOSE_LOGGING=true to enable detailed logs.")


def log_system_info():
    """Log detailed system information."""
    logging.info(f"Platform: {platform.platform()}")
    logging.info(f"Python version: {platform.python_version()}")

    # Only log detailed system info in verbose mode
    if not VERBOSE_LOGGING:
        return

    # CPU info
    if hasattr(psutil, "cpu_count"):
        logging.debug(
            f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")

    # Memory info
    mem = psutil.virtual_memory()
    logging.debug(
        f"System memory: Total={mem.total/(1024**3):.2f}GB, Available={mem.available/(1024**3):.2f}GB ({mem.percent}% used)")

    # GPU info
    if torch.cuda.is_available():
        cuda_version = torch.__version__ if hasattr(
            torch, "__version__") else "Unknown"
        logging.debug(f"PyTorch version: {cuda_version}")
        device_count = torch.cuda.device_count()
        logging.debug(f"GPU devices available: {device_count}")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            logging.debug(
                f"  GPU {i}: {props.name}, {props.total_memory/(1024**3):.2f}GB memory")


load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("RAG_API_KEY", secrets.token_urlsafe(32))
API_KEY_NAME = "X-API-Key"
INACTIVITY_TIMEOUT = int(os.getenv("INACTIVITY_TIMEOUT", 600))


# --- Subprocess Manager ---
class RetrieverProcessManager:
    """Manages the lifecycle of the retriever subprocess."""

    def __init__(self):
        self.process = None
        self.request_queue = None
        self.response_queue = None
        self.last_active_time = None
        self.lock = threading.Lock()

    def _start_worker(self):
        print("[Manager] Starting new retriever subprocess...")
        self.request_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(
            target=retriever_worker,
            args=(self.request_queue, self.response_queue),
            daemon=True
        )
        self.process.start()
        self.last_active_time = time.time()

    def _stop_worker(self):
        if self.process and self.process.is_alive():
            print("[Manager] Stopping retriever subprocess...")
            try:
                if self.request_queue:
                    self.request_queue.put(None)
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.terminate()
            except Exception as e:
                print(f"[Manager] Error stopping worker: {e}")
        self.process = None
        self.request_queue = None
        self.response_queue = None

    @contextmanager
    def get_worker(self):
        """A thread-safe context manager to ensure the worker is running."""
        with self.lock:
            if not (self.process and self.process.is_alive()):
                self._start_worker()
            self.last_active_time = time.time()

        if self.request_queue is None or self.response_queue is None:
            raise RuntimeError("Failed to get valid worker queues.")

        yield self.request_queue, self.response_queue


# --- Background Inactivity Monitor ---
def inactivity_monitor_task(manager: RetrieverProcessManager, stop_event: threading.Event):
    """Runs in a background thread to stop the worker after a timeout."""
    print(
        f"[Monitor] Inactivity monitor started. Timeout set to {INACTIVITY_TIMEOUT} seconds.")
    while not stop_event.wait(30):
        with manager.lock:
            if (manager.process and
                manager.process.is_alive() and
                manager.last_active_time and
                    (time.time() - manager.last_active_time > INACTIVITY_TIMEOUT)):
                print(
                    f"[Monitor] Worker inactive for more than {INACTIVITY_TIMEOUT}s. Stopping.")
                manager._stop_worker()


# --- FastAPI Application ---
retriever_manager = RetrieverProcessManager()
monitor_stop_event = threading.Event()

app = FastAPI(
    title="RAG Retriever API",
    description="API that manages a retriever subprocess with inactivity monitoring.",
    version="2.3.0",
)

# Add request logging middleware


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    start_time = time.time()

    # Only log request start in verbose mode
    if VERBOSE_LOGGING:
        logging.debug(
            f"[{request_id}] Request started: {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        log_level = logging.DEBUG if VERBOSE_LOGGING else logging.INFO
        # Use appropriate level based on response status
        if response.status_code >= 400:
            logging.warning(
                f"[{request_id}] {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
        elif VERBOSE_LOGGING:
            logging.debug(
                f"[{request_id}] Request completed: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")

        return response
    except Exception as e:
        process_time = time.time() - start_time
        logging.error(
            f"[{request_id}] Request failed: {request.method} {request.url.path} - Error: {str(e)} - Time: {process_time:.3f}s")

        # Only log traceback in verbose mode
        if VERBOSE_LOGGING:
            logging.error(
                f"[{request_id}] Exception traceback: {traceback.format_exc()}")

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Internal server error: {str(e)}"},
        )

# Exception handler for unhandled exceptions


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {str(exc)}")

    # Only log traceback in verbose mode
    if VERBOSE_LOGGING:
        logging.error(f"Exception traceback: {traceback.format_exc()}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred. Check server logs for details."}
    )


@app.on_event("startup")
def startup_event():
    logging.info("==== RAG Retriever API Starting ====")
    log_system_info()

    monitor_thread = threading.Thread(
        target=inactivity_monitor_task,
        args=(retriever_manager, monitor_stop_event),
        daemon=True
    )
    monitor_thread.start()
    logging.info("Inactivity monitor thread started")


@app.on_event("shutdown")
def shutdown_event():
    print("[Manager] Application shutting down.")
    monitor_stop_event.set()
    retriever_manager._stop_worker()


class RetrievalRequest(BaseModel):
    query: str = Field(..., description="The query to search for")
    top_k: int = Field(5, description="Number of results to return")
    collection: Optional[str] = Field(
        None, description="Force a specific collection")
    debug: bool = Field(False, description="Enable debug mode")


async def get_api_key(api_key: str = Header(None, alias=API_KEY_NAME)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key"
        )
    return api_key


@app.get("/", tags=["Health"])
async def health_check():
    """Provides the health status of the API and its worker process."""
    response = {
        "status": "healthy",
        "message": "RAG Retriever API is running",
    }
    if not (retriever_manager.process and retriever_manager.process.is_alive()):
        response["worker_status"] = "inactive"
        response["worker_device"] = "N/A"
        return response

    try:
        with retriever_manager.get_worker() as (req_q, res_q):
            await asyncio.to_thread(req_q.put, {"type": "health_check"})
            worker_health = await asyncio.to_thread(res_q.get)

            response["worker_status"] = worker_health.get("status", "unknown")
            device = worker_health.get("device")
            device_name = worker_health.get("device_name")
            response["worker_device"] = f"{device} ({device_name})" if device == "cuda" else device
    except Exception as e:
        response["worker_status"] = "unresponsive"
        response["worker_device"] = f"Error communicating with worker: {e}"

    return response


@app.post("/retrieve", tags=["Retrieval"])
async def retrieve(request: RetrievalRequest, api_key: str = Depends(get_api_key)) -> Dict[str, Any]:
    """Sends a query to the retriever worker and returns the results."""
    request_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    start_time = time.time()

    # Basic info logging is always enabled
    if VERBOSE_LOGGING:
        logging.debug(
            f"[{request_id}] New retrieve request: query='{request.query[:50]}{'...' if len(request.query) > 50 else ''}', top_k={request.top_k}")
    else:
        logging.info(
            f"Processing query (length: {len(request.query)}, top_k: {request.top_k})")

    # Only log memory status in verbose mode
    if VERBOSE_LOGGING:
        # Log memory status at request start
        mem = psutil.virtual_memory()
        logging.debug(
            f"[{request_id}] System memory at request start: {mem.available/(1024**3):.2f}GB available ({mem.percent}% used)")

        if torch.cuda.is_available():
            # Log GPU memory at request start
            for i in range(torch.cuda.device_count()):
                free_mem = torch.cuda.memory_reserved(
                    i) - torch.cuda.memory_allocated(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory
                logging.debug(
                    f"[{request_id}] GPU {i} memory at request start: {free_mem/(1024**3):.2f}GB free of {total_mem/(1024**3):.2f}GB total")

    try:
        with retriever_manager.get_worker() as (req_q, res_q):
            if VERBOSE_LOGGING:
                logging.debug(
                    f"[{request_id}] Worker acquired, sending request to worker")
            worker_payload = request.dict()
            await asyncio.to_thread(req_q.put, worker_payload)

            if VERBOSE_LOGGING:
                logging.debug(f"[{request_id}] Waiting for worker response")
            results = await asyncio.to_thread(res_q.get)
            if VERBOSE_LOGGING:
                logging.debug(f"[{request_id}] Received response from worker")

        if "error" in results:
            error_type = results.get("error")
            error_message = results.get(
                "message", "An unknown error occurred in the worker.")

            logging.error(
                f"[{request_id}] Worker returned error: {error_type} - {error_message}")

            if error_type == "OutOfMemoryError":
                # Always log detailed memory info on OOM error, even without verbose mode
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

                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_message)
            else:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    detail=f"Error in worker: {error_message}")

        result_count = len(results.get("results", []))
        elapsed = time.time() - start_time

        # Always log basic completion info
        logging.info(
            f"Request completed in {elapsed:.2f}s, returned {result_count} results")

        # Additional details only in verbose mode
        if VERBOSE_LOGGING:
            logging.debug(
                f"[{request_id}] Request details: query='{request.query[:30]}{'...' if len(request.query) > 30 else ''}'")
            if "query_info" in results:
                query_info = results["query_info"]
                logging.debug(f"[{request_id}] Query analysis: search_type={query_info.get('search_type')}, " +
                              f"detected_topics={query_info.get('detected_topics')}, " +
                              f"detected_locations={query_info.get('detected_locations')}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return results

    except Exception as e:
        elapsed = time.time() - start_time
        logging.error(
            f"[{request_id}] Request failed after {elapsed:.2f}s: {str(e)}")

        # Only log full traceback in verbose mode
        if VERBOSE_LOGGING:
            logging.error(
                f"[{request_id}] Exception traceback: {traceback.format_exc()}")

        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error processing request: {str(e)}")

    finally:
        # Clean up regardless of success or failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Log memory cleanup in verbose mode only
        if VERBOSE_LOGGING:
            mem = psutil.virtual_memory()
            logging.debug(
                f"[{request_id}] Final system memory: {mem.available/(1024**3):.2f}GB available ({mem.percent}% used)")


if __name__ == "__main__":
    # The set_start_method call is removed from here as it's now at the top level.
    print(f"Using API key: {API_KEY}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
