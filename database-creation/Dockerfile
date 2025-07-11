# Dockerfile (Corrected)

# --- Build Stage ---
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python, pip, and the venv package
# --- THIS IS THE CORRECTED LINE ---
RUN apt-get update && \
    apt-get install -y python3.12 python3-pip python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

# Set a virtual environment for a clean package installation
WORKDIR /app
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements file first to leverage Docker caching
COPY requirements-api.txt .

# Install PyTorch and other requirements into the virtual environment
# Corrected to use the standard CUDA 12.1 wheel
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir -r requirements-api.txt


# --- Final Stage ---
# Use the lean runtime image for the final product
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python runtime
RUN apt-get update && \
    apt-get install -y python3.12 && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

# Copy the application code
COPY . .

# Set the PATH to use the venv
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port [cite: 3]
EXPOSE 8000

# Define the command to run your application [cite: 3]
CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8000"]