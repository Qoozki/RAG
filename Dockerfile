FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Python source
COPY app.py config.py database.py lang_utils.py main.py rag_service.py rate_limiter.py vector_store.py ./

# app.py expects the HTML files inside ./templates
COPY index.html login.html ./templates/

# Persistent data (PDFs + local Qdrant DB)
RUN mkdir -p /app/docs /app/qdrant_db
VOLUME ["/app/docs", "/app/qdrant_db"]

EXPOSE 8000

# Bind to 0.0.0.0 (main.py binds to 127.0.0.1 which is unreachable from outside the container).
# Keep a single worker: local-mode Qdrant only allows one process to open the DB.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-timeout", "900"]
