# Stage 1: The Builder
# Use the official python-slim image as a lean base with Python pre-installed.
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1
ENV HF_HOME="/app/cache/huggingface"
ENV VENV_PATH="/opt/venv"
# Add VENV_PATH to the PATH
ENV PATH="$VENV_PATH/bin:$PATH"

WORKDIR /app

# Install only build-essential for any C extensions, then create the venv.
# Python is already in the base image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m venv $VENV_PATH

# Install Python dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r requirements-api.txt

# Copy application code and cache models
COPY rag_api.py rag_retriever_worker.py cache_models.py ./
RUN python cache_models.py && \
    # Clean up the temporary download cache
    rm -rf ${HF_HOME}/xet

#-----------------------------------------------------------------------------------

# Stage 2: The Final Image
# Start from the same lean python-slim base.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV HF_HOME="/app/cache/huggingface"
ENV VENV_PATH="/opt/venv"
ENV PATH="$VENV_PATH/bin:$PATH"

# Install only the essential shared libraries needed by torch/numpy at runtime.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual environment, model cache, and application code from the builder
COPY --from=builder ${VENV_PATH} ${VENV_PATH}
COPY --from=builder ${HF_HOME} ${HF_HOME}
WORKDIR /app
COPY --from=builder /app /app

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run your application [cite: 3]
CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8000"]