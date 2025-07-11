# ==============================================================================
# SCRIPT: rag_api.py
#
# PURPOSE:
#   - Provides a FastAPI-based API for the RAG retriever
#   - Uses API key authentication for secure access
#   - Always connects to Qdrant on localhost
#   - Exposes the retrieval functionality for external applications
#
# HOW TO USE:
#   1. Set your API key in the .env file or as an environment variable
#   2. Run the server: uvicorn rag_api:app --host 0.0.0.0 --port 8000
#   3. Make requests to the API with your API key in the header
#
# EXAMPLE CURL REQUEST:
#   curl -X POST "http://localhost:8000/retrieve"
#       -H "Content-Type: application/json"
#       -H "X-API-Key: your-api-key"
#       -d '{"query": "Wo finde ich psychologische Hilfe in Siegen?", "top_k": 3}'
# ==============================================================================
import os
import secrets
import torch
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Import the RAG retriever
from rag_retriever import retrieve_content

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("RAG_API_KEY", secrets.token_urlsafe(32))
API_KEY_NAME = "X-API-Key"

# Fixed Qdrant settings for localhost
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Initialize FastAPI app
app = FastAPI(
    title="RAG Retriever API",
    description="API for retrieving information from medical guidelines and location databases",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key security scheme
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


# API Models
class RetrievalRequest(BaseModel):
    query: str = Field(..., description="The query to search for")
    top_k: int = Field(5, description="Number of results to return")
    collection: Optional[str] = Field(
        None, description="Force a specific collection (guideline, location, or both)")
    debug: bool = Field(False, description="Enable debug mode")


class ErrorResponse(BaseModel):
    detail: str


# Authentication dependency
async def get_api_key(api_key: str = Header(None, alias=API_KEY_NAME)):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
        )
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return api_key


# API Routes
@app.get("/", tags=["Health"])
async def health_check():
    """Check if the API is running"""
    # Check if CUDA is available
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    device_info = f"Using {device}"
    if device == "CUDA":
        device_info += f" ({torch.cuda.get_device_name(0)})"

    return {"status": "healthy", "message": "RAG Retriever API is running", "device": device_info}


@app.post("/retrieve", tags=["Retrieval"])
async def retrieve(request: RetrievalRequest, api_key: str = Depends(get_api_key)) -> Dict[str, Any]:
    """
    Retrieve content from the RAG system based on the query

    This endpoint searches medical guidelines and location databases 
    based on the query and returns relevant results.
    """
    try:
        results = retrieve_content(
            query=request.query,
            top_k=request.top_k,
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            force_collection=request.collection,
            debug=request.debug
        )
        return results
    except torch.cuda.OutOfMemoryError:
        # Handle CUDA out of memory error specifically
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GPU memory exhausted. Try reducing batch size or using CPU mode.",
        )
    except RuntimeError as e:
        # Handle other CUDA-related errors
        if "CUDA" in str(e):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"CUDA error: {str(e)}. The system will use CPU instead.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Runtime error: {str(e)}",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving content: {str(e)}",
        )


@app.get("/api-key", tags=["Admin"])
async def generate_api_key():
    """Generate a new API key (only for demonstration purposes)"""
    # In a production environment, this should be properly secured
    # and only accessible to administrators
    new_key = secrets.token_urlsafe(32)
    return {"new_api_key": new_key,
            "message": "This is only for demonstration. In production, API keys should be securely managed."}


# Run the server if executed directly
if __name__ == "__main__":
    # Print the API key for development convenience
    print(f"Using API key: {API_KEY}")
    print("Start the server with: uvicorn rag_api:app --host 0.0.0.0 --port 8000")

    # Alternatively, you can run directly from here (not recommended for production)
    # uvicorn.run(app, host="0.0.0.0", port=8000)
