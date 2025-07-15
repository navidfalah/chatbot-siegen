import os
import secrets
import torch
import time
import multiprocessing
import asyncio
import threading
from contextlib import contextmanager
from typing import Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, Header, status
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

from rag_retriever_worker import retriever_worker

# This is the crucial change.
# Set the start method at the module level to ensure it's set when Uvicorn imports this file.
multiprocessing.set_start_method("spawn", force=True)

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


@app.on_event("startup")
def startup_event():
    monitor_thread = threading.Thread(
        target=inactivity_monitor_task,
        args=(retriever_manager, monitor_stop_event),
        daemon=True
    )
    monitor_thread.start()


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
    try:
        with retriever_manager.get_worker() as (req_q, res_q):
            worker_payload = request.dict()
            await asyncio.to_thread(req_q.put, worker_payload)
            results = await asyncio.to_thread(res_q.get)

        if "error" in results:
            error_type = results.get("error")
            error_message = results.get(
                "message", "An unknown error occurred in the worker.")

            if error_type == "OutOfMemoryError":
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_message)
            else:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    detail=f"Error in worker: {error_message}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error processing request: {str(e)}")


if __name__ == "__main__":
    # The set_start_method call is removed from here as it's now at the top level.
    print(f"Using API key: {API_KEY}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
