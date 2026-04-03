FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    fastapi==0.115.0 uvicorn[standard]==0.32.0 \
    aiokafka==0.11.0 redis[hiredis]==5.2.1
COPY api-gateway/src/ .
COPY agent-orchestrator/proto_codec.py ./proto_codec.py
COPY agent-orchestrator/causal_engine.py ./causal_engine.py
COPY agent-orchestrator/svar_validation.py ./svar_validation.py
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
