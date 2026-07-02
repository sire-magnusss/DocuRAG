# ---------------------------------------------------------------------------
# DocuRAG / pyLLMSearch  –  CPU-based production image
#
# For GPU support swap the base image for one that includes CUDA, e.g.:
#   FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
# and install Python 3.11 manually, or use a pytorch base image.
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# System dependencies needed by some ML libraries (e.g. MuPDF, tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the package definition first so Docker can cache the pip install layer
COPY pyproject.toml ./
COPY src/ src/

# Install the package in non-editable mode (no dev extras needed at runtime)
RUN pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Runtime configuration
#
# Mount your YAML config files into /config at `docker run` time, e.g.:
#   -v /host/path/rag_config.yaml:/config/rag_config.yaml
#   -v /host/path/llm_config.yaml:/config/llm_config.yaml
#
# Required env vars (override via -e or docker-compose environment section):
#   FASTAPI_RAG_CONFIG  – path inside the container to the RAG config YAML
#   FASTAPI_LLM_CONFIG  – path inside the container to the LLM config YAML
#
# Optional env vars:
#   LLMSEARCH_API_KEY   – when set, every request must include X-Api-Key header
#   OPENAI_API_KEY      – required when using OpenAI models
#   GOOGLE_API_KEY      – required when using Gemini image parsing
# ---------------------------------------------------------------------------

ENV FASTAPI_RAG_CONFIG=/config/rag_config.yaml
ENV FASTAPI_LLM_CONFIG=/config/llm_config.yaml

EXPOSE 8000

CMD ["llmsearchapi"]
