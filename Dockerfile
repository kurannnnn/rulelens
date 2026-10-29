FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV FASTEMBED_CACHE_PATH=/app/.fastembed

# 1. Install packages
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 2. Copy code and data (not .env, see .dockerignore)
COPY src ./src
COPY data ./data
COPY output/chunks.jsonl ./output/chunks.jsonl

# 3. Build the search index and download the models once, at build time
RUN python src/build_index.py && \
    python -c "import sys; sys.path.insert(0, 'src'); from search import HybridSearcher; HybridSearcher(rerank=True)"

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]