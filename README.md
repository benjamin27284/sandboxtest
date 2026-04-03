# ABMS Market Simulation

ABMS is an agent-based market simulation project built around a continuous double auction limit order book.

This repository contains two related implementations:

- A root-level Python prototype for a local multi-agent limit order book simulation
- A larger distributed stack with a C++ matching engine, Python orchestrator, FastAPI gateway, Kafka, Redis, Qdrant, and a Next.js dashboard
- A larger distributed stack with a C++ matching engine, a Ray actor-based Python orchestrator, FastAPI gateway, Kafka, Redis, Qdrant, and a Next.js dashboard

## What Is In The Repo

### Root-level prototype

The top-level Python files provide a compact local simulation:

- `lob.py`: heap-based limit order book with price-time priority
- `agents.py`: several rule-based trader types
- `llm_agent.py`: LLM-driven fundamental agent using the DashScope OpenAI-compatible API
- `simulation.py`: tick-based simulation runner and summary output

This path is useful for quickly exercising the core matching and agent logic without bringing up the full platform.

### Distributed platform

The `services/` and `frontend/` folders contain the full multi-service system:

- `services/lob-engine`: C++ Kafka-connected matching engine
- `services/agent-orchestrator`: Ray-based simulation orchestration, event impact analysis, Python order book, proto codec, and validation logic
- `services/agent-orchestrator`: Ray actor-based simulation orchestration, event impact analysis, Python order book, proto codec, and validation logic
- `services/api-gateway`: FastAPI service exposing REST endpoints and a WebSocket market-data bridge
- `frontend`: Next.js dashboard for live charts, order book depth, controls, and report export
- `proto/market.proto`: protobuf schema shared across services
- `docker-compose.yml`: local orchestration for the stack

## Main Features

- Continuous double auction matching with price-time FIFO priority
- Multiple agent styles, including noise, momentum, market-making, and LLM-guided behavior
- Ray actor-based parallel agent orchestration for large-scale simulation workloads
- Multiple agent styles, including noise, momentum, market-making, and LLM-guided behavior
- Kafka-based protobuf message flow between services
- Real-time market streaming over WebSocket to the dashboard
- Event impact analysis that runs per-asset mini simulations in Python order books
- Shock injection and simulation control via the API gateway

## Quick Start

### Option 1: run the root-level Python prototype

1. Create a virtual environment and install the root Python dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Copy the example environment file and add your DashScope credentials:

```powershell
Copy-Item .env.example .env
```

3. Run the simulation:

```powershell
python simulation.py --ticks 50
```

Notes:

- `llm_agent.py` requires `DASHSCOPE_API_KEY` at import time
- `requirements.txt` only covers the lightweight root prototype, not the full distributed stack

### Option 2: run the distributed stack with Docker Compose

The compose file starts:

- Zookeeper
- Kafka
- Redis
- Qdrant
- C++ LOB engine
- Python agent orchestrator
- FastAPI API gateway
- Next.js frontend

Before starting, provide environment values such as:

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`

Then run:

```powershell
docker compose up --build
```

Default local endpoints:

- Frontend: `http://localhost:3000`
- API gateway: `http://localhost:8000`
- WebSocket: `ws://localhost:8000/ws/market`

## Testing

The repository includes lightweight tests for the Python matching engine and protobuf contract:

```powershell
python tests\test_matching_engine.py
python tests\test_proto_contract.py
```

These focus on:

- crossing and non-crossing orders
- partial fills
- price-time priority
- passive-price execution semantics
- protobuf encode/decode compatibility for service integration

## API And UI Notes

The FastAPI gateway includes endpoints for:

- health checks
- simulation start and stop control
- simulation status
- shock injection
- report export

The frontend dashboard includes:

- real-time price charting
- order book depth display
- event-analysis visualization
- simulation controls
- EGCIRF JSON export

## Project Structure

```text
.
|-- agents.py
|-- llm_agent.py
|-- lob.py
|-- simulation.py
|-- proto/
|   `-- market.proto
|-- services/
|   |-- agent-orchestrator/
|   |-- api-gateway/
|   `-- lob-engine/
|-- frontend/
|-- tests/
|-- docker-compose.yml
`-- requirements.txt
```

## Environment Variables

Example values are provided in `.env.example`:

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `EODHD_API_KEY`

Additional service-level settings are defined in `docker-compose.yml` and the service code.

## Current Caveats

- The root-level simulation imports the LLM agent directly, so a missing DashScope key will stop startup
- Full-stack dependencies are defined mostly in Dockerfiles and service code, not in the top-level `requirements.txt`
- Some comments and strings in source files still contain legacy encoding artifacts, even though this README has been cleaned up
