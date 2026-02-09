# Stage 1: Build the Next.js static files
FROM node:22-alpine AS frontend-builder

WORKDIR /app

# Copy package files first (for better caching)
COPY package*.json ./
RUN npm ci

# Copy all frontend files
COPY . .

# Build the Next.js app (creates 'out' directory with static files)
RUN npm run build

# Stage 2: Create the final Python container
# FROM python:3.12-slim
FROM python:3.12-slim

# Copy uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml uv.lock ./

# Set the PATH to include the virtual environment's bin directory
ENV PATH="/app/.venv/bin:$PATH"
# Optionally, set the VIRTUAL_ENV variable
ENV VIRTUAL_ENV="/app/.venv"

# Install project dependencies using uv.
# The --frozen flag ensures uv uses the lock file.
# The --no-install-project flag installs dependencies without the project's source code [6, 9, 14].
# RUN uv pip install --frozen --no-install-project
RUN uv sync --frozen --no-cache

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Create directory for Azure Files mount (ChromaDB persistence)
RUN mkdir -p /data/chromadb && chmod 777 /data/chromadb

# Copy the FastAPI server
COPY api/index.py .
COPY api/pipeline.py .
COPY api/vectorstore.py .

# Copy the Next.js static export from builder stage
COPY --from=frontend-builder /app/out ./static

# Health check using curl (faster and more reliable than Python)
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose port 8000 (FastAPI will serve everything)
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "8000"]