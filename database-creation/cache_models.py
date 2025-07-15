import torch
from rag_retriever_worker import initialize_retrieval_components


def preload_models():
    """
    Initializes and caches all required models to the local Hugging Face cache.
    """
    print("Starting model pre-loading...")
    try:
        # Determine the device, defaulting to CPU for the build process
        # The actual runtime device will be determined by the worker
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Building on device: {device}")

        # This call will download the models to the default cache location
        initialize_retrieval_components(device=device)

        print("✅ Models have been successfully pre-loaded and cached.")
    except Exception as e:
        print(f"❌ An error occurred during model pre-loading: {e}")
        # Exit with a non-zero status to fail the Docker build if models can't be downloaded
        exit(1)


if __name__ == "__main__":
    preload_models()
