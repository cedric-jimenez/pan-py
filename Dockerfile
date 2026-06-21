# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and healthcheck
RUN apt-get update && apt-get install -y \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set default PORT (Railway will override this)
ENV PORT=8000

# Expose port
EXPOSE ${PORT}

# Health check (using curl with PORT env variable)
# Note: start-period covers YOLO load (~10-20s) + DINOv2 load + inference warmup
# (a dummy forward pass primes oneDNN caches, ~10-15s) before traffic is routed.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application with JSON logging
# Use shell form to allow environment variable substitution
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --log-config app/uvicorn_logging_config.json
