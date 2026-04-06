# =============================================================================
# Agent Orchestrator — Ray + Python
# =============================================================================
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    ray[default]==2.38.0 \
    aiokafka==0.11.0 \
    redis[hiredis]==5.2.1 \
    qdrant-client==1.12.1 \
    openai==1.55.0 \
    sentence-transformers==3.3.1 \
    numpy==2.1.3

COPY . .

CMD ["python", "-m", "orchestrator"]
