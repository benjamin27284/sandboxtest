# ABMS — Agent-Based Market Simulation Platform

A distributed simulation where **1,000 LLM-powered trading agents** interact through a **Continuous Double Auction Limit Order Book**. When news events occur, each impacted asset is independently simulated by heterogeneous agents trading in dedicated order books — prices emerge from agent interaction, not equations.

---

## Architecture

```
                     ┌──────────────────────────────────┐
                     │       Next.js 15 Dashboard        │
                     │  Price chart · Multi-asset sim    │
                     │  Shock injector · EGCIRF export   │
                     └──────────────┬───────────────────┘
                                    │ WebSocket (JSON)
                     ┌──────────────┴───────────────────┐
                     │      FastAPI API Gateway          │
                     │  REST · WS broadcast · NL query   │
                     │  Event analysis (background task) │
                     └──────────────┬───────────────────┘
                                    │ Kafka (protobuf)
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
 ┌─────────┴──────────┐   ┌────────┴─────────┐   ┌─────────┴──────────┐
 │  Agent Orchestrator │   │   C++ LOB Engine  │   │   Infrastructure   │
 │  1,000 Ray Actors   │──▶│  854K orders/sec  │   │  Redis · Qdrant    │
 │  Python · async     │◀──│  Price-time FIFO  │   │  Kafka · Zookeeper │
 └─────────┬──────────┘   └───────────────────┘   └────────────────────┘
           │
 ┌─────────┴───────────────────────────────────────────┐
 │         Three-Tier Cognitive Architecture            │
 │  Tier 1: Redis       Quantitative state (cash, VaR) │
 │  Tier 2: Qdrant      Semantic RAG memory             │
 │  Tier 3: SLM         Episodic summaries              │
 │  Shield: QuantEngine Bayesian · BL · VaR · DCF · CPT│
 └──────────────────────────────────────────────────────┘
```

---

## Quick Start

### Full Stack (Docker)

```bash
# 1. Configure environment
cp .env.txt .env
# Edit .env — add DASHSCOPE_API_KEY and EODHD_API_KEY

# 2. Start all services
docker compose up -d

# 3. Open dashboard
http://localhost:3000/dashboard

# 4. Teardown
docker compose down -v
```

### Python Prototype Only

```bash
pip install -r requirements.txt
python simulation.py --ticks 50
```

### Event Impact Analysis (Standalone)

```bash
cd services/agent-orchestrator

# Offline demo (no API key needed — uses cached LLM response)
python run_event_analysis.py --offline

# Online (requires DASHSCOPE_API_KEY)
set DASHSCOPE_API_KEY=your-key
set DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
python run_event_analysis.py
```

### C++ LOB Benchmark

```bash
cd services/lob-engine
cmake -B out -DCMAKE_BUILD_TYPE=Release
cmake --build out --parallel
./out/lob_bench
```

Expected:
```
[PASS] test_basic_match
[PASS] test_partial_fill_multi_level
[PASS] test_market_order
[PASS] test_cancel
[PASS] test_get_mid_price
[BENCH] 2000000 orders in 2342.1 ms = 853952 orders/sec
```

---

## Services

### C++ LOB Engine (`services/lob-engine`)

Ultra-low-latency matching engine.

- **Data structure:** `std::map<double, std::deque<Order>>` for bids/asks
- **Matching:** price-time FIFO — fills at passive order's price
- **Cancel:** zero-allocation lazy tombstone (`remaining = 0`)
- **Throughput:** ≥854K orders/sec on a single core (`-O3 -march=native`)
- **Kafka:** consumes `orders_submit`, `orders_cancel` → produces `executions`, `market_data`

### Agent Orchestrator (`services/agent-orchestrator`)

Runs 1,000 heterogeneous LLM-powered trading agents via Ray.

**Agent Personas** (weighted by daily trading volume):

| Persona | Volume | Sophistication | Shock Behavior |
|---------|--------|----------------|----------------|
| HFT Market Maker | 45% | 1.2× | Widens spreads 3–5×, pulls liquidity |
| Momentum Quant | 25% | 1.1× | Chases directional momentum |
| Macro Event-Driven | 15% | 1.0× | Reprices fundamentals aggressively |
| Retail Sentiment | 10% | 2.5× | Delayed mean-reversion |
| Passive Index Fund | 5% | 0.5× | Minimal intraday participation |

**Execution Strategies** (assigned round-robin, 25% each):

| Strategy | Behavior |
|----------|---------|
| Aggressive | Market order + slippage, fills immediately |
| Passive | Limit order at the spread, rests in book |
| TWAP | 3-tranche order slicing over time |
| DDQL | Double Deep Q-Learning (learns from experience) |

**Three-Tier Cognitive Architecture:**

1. **Redis (Tier 1)** — quantitative snapshot: cash, inventory, PnL, VaR, drawdown
2. **Qdrant (Tier 2)** — RAG retrieval of semantically similar past events
3. **SLM (Tier 3)** — qwen-turbo episodic summarization of recent ticks

**Deterministic Math Shield** (LLM never computes numbers):

| Module | Method |
|--------|--------|
| Bayesian Update | Conjugate Gaussian posterior |
| Black-Litterman | Optimal position sizing |
| VaR | Parametric 95% confidence |
| DCF | Multi-stage Gordon Growth valuation |
| CPT | Kahneman-Tversky loss aversion signal |

**MASS Signal** — emergent consensus across all 1,000 agents per tick:
```
Signal = 0.6 × mean_direction_score - 0.4 × stdev_direction_score
```

### API Gateway (`services/api-gateway`)

FastAPI server that bridges Kafka ↔ WebSocket and exposes REST endpoints.

### Frontend (`frontend`)

Next.js 15 real-time dashboard with:
- Live price chart (WebSocket)
- Multi-asset event impact chart (Redis polling)
- Shock injector with preset events and custom headline input
- Telemetry bar: Mid Price, Spread, Consensus, MASS Signal, Volume
- EGCIRF report export

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + WS connection count |
| POST | `/api/simulation/control` | Start / stop simulation |
| GET | `/api/simulation/status` | Current simulation state |
| POST | `/api/event-analysis/run` | Run LLM analysis + multi-asset simulation (background) |
| GET | `/api/event-analysis/latest` | Fetch latest simulation results |
| GET | `/api/event-analysis/status` | Check if analysis is running |
| POST | `/api/shocks/inject` | Inject shock via Kafka → Orchestrator |
| GET | `/api/reports/egcirf` | Counterfactual impulse response report |
| POST | `/api/reports/svar` | Guerini-Moneta SVAR validation |
| POST | `/api/nl/query` | Natural language analyst query |
| WS | `/ws/market` | Real-time market telemetry stream |

**WebSocket message format:**
```json
{
  "topic": "market_data | tick_summary | executions",
  "data": { ... },
  "timestamp": 1712345678.9
}
```

---

## Event Impact Analysis

When an event is injected from the dashboard, the pipeline runs:

```
Event Headline (from UI or run_event_analysis.py)
    │
    ▼
LLM Analyst (qwen-plus)
  → Identifies 3–5 impacted assets
  → Assigns direction (up/down), magnitude (high/medium/low), confidence
    │
    ▼
Per-Asset Agent-Driven LOB Simulation (runs in parallel background task)
  For EACH asset:
    - 4 heterogeneous agents (HFT, Momentum, Macro, Retail)
    - Each agent calls LLM → buy/sell/hold decision
    - Orders submitted to dedicated Python PyOrderBook
    - Price-time FIFO matching → price trajectory
    - 10 simulation ticks
    │
    ▼
Results stored in Redis → Frontend polls and renders MultiAssetChart
```

---

## Causal Inference (EGCIRF)

Pearl's do-calculus over a 7-node structural causal model:

```
interest_rate (exogenous)
    ├──▶ inflation = 0.03 - 0.4×rate
    └──▶ gdp_growth = 0.025 - 0.3×rate + 0.1×inflation
              └──▶ market_sentiment = 0.5×gdp - 0.3×inflation
                        ├──▶ liquidity = -0.5×rate + 0.4×sentiment
                        └──▶ volatility = 0.15 - 0.2×sentiment - 0.1×liquidity
                                  └──▶ asset_price = 100 + 20×sentiment
                                                   + 10×liquidity - 5×volatility
```

**EGCIRF** runs `do(variable = value)` (forcing a node, cutting parent edges) across N Monte Carlo runs and computes the mean impulse response trajectory vs. baseline.

Example query via natural language:
```
POST /api/nl/query
{"query": "inject 200bps rate hike and show EGCIRF"}
```

---

## Kafka Topics

All topics use Protocol Buffers 3 wire format (schema: `proto/market.proto`).

| Topic | Partitions | Direction | Message |
|-------|-----------|-----------|---------|
| `orders_submit` | 8 | Orchestrator → LOB | Order |
| `orders_cancel` | 4 | Orchestrator → LOB | CancelOrder |
| `executions` | 8 | LOB → Orchestrator, Gateway | Execution |
| `market_data` | 4 | LOB → Gateway | MarketSnapshot |
| `tick_summary` | 1 | Orchestrator → Gateway | TickSummary |
| `exogenous_shocks` | 1 | Gateway → Orchestrator | ExogenousShock (JSON) |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DASHSCOPE_API_KEY` | — | Aliyun DashScope key (required) |
| `DASHSCOPE_BASE_URL` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | LLM endpoint |
| `DASHSCOPE_API_KEYS` | — | Comma-separated keys for rate-limit distribution |
| `EODHD_API_KEY` | — | Market data key (for SVAR validation) |
| `KAFKA_BROKERS` | `localhost:9092` | Kafka bootstrap servers |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6334` | Qdrant gRPC port |
| `NUM_AGENTS` | `1000` | Number of Ray trading agents |
| `PRIMARY_LLM_MODEL` | `qwen-plus` | Main LLM model |
| `SLM_MODEL` | `qwen-turbo` | Small model for episodic summaries |
| `RAY_NUM_CPUS` | `8` | Ray cluster CPU allocation |
| `LLM_TIMEOUT_SECONDS` | `15` | LLM call timeout |
| `TICKS_PER_SUMMARY` | `10` | Ticks between MASS signal summaries |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000/ws/market` | Frontend WebSocket URL |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend API URL |
| `CORS_ORIGINS` | `http://localhost:3000` | CORS allowed origins |

---

## Project Structure

```
.
├── proto/
│   └── market.proto                # Protobuf schema (source of truth)
│
├── services/
│   ├── lob-engine/                 # C++20 matching engine
│   │   ├── include/order.h         # Order + Execution structs
│   │   ├── include/orderbook.h     # OrderBook class
│   │   ├── src/orderbook.cpp       # Price-time FIFO matching
│   │   ├── src/main.cpp            # Kafka consumer/producer
│   │   ├── tests/bench_orderbook.cpp
│   │   └── CMakeLists.txt
│   │
│   ├── agent-orchestrator/
│   │   ├── actors/base_actor.py    # TradingAgentActor (3-tier cognitive)
│   │   ├── memory/
│   │   │   ├── state_store.py      # Redis state (Tier 1)
│   │   │   ├── semantic_memory.py  # Qdrant RAG (Tier 2)
│   │   │   └── episodic_buffer.py  # SLM episodic (Tier 3)
│   │   ├── math_engine/
│   │   │   └── quant_models.py     # Bayesian, BL, VaR, DCF, CPT
│   │   ├── config/settings.py      # Environment config
│   │   ├── orchestrator.py         # Main tick loop + MASS signal
│   │   ├── matching_engine.py      # Python LOB for event simulations
│   │   ├── event_impact_analyzer.py# LLM analysis + per-asset simulation
│   │   ├── causal_engine.py        # Pearl's SCM + EGCIRF
│   │   ├── ddql_agent.py           # Double DQN execution agent
│   │   ├── svar_validation.py      # SVAR empirical validation
│   │   ├── proto_codec.py          # Pure-Python protobuf codec
│   │   └── run_event_analysis.py   # Standalone event analysis CLI
│   │
│   └── api-gateway/
│       └── src/main.py             # FastAPI + WebSocket bridge
│
├── frontend/
│   └── src/
│       ├── app/dashboard/page.tsx  # Main dashboard page
│       ├── components/
│       │   ├── charts/
│       │   │   ├── PriceChart.tsx       # Live price time-series
│       │   │   └── MultiAssetChart.tsx  # Event impact chart
│       │   └── controls/
│       │       └── ShockInjector.tsx    # Event injection panel
│       ├── hooks/useMarketSocket.ts # WebSocket hook + auto-reconnect
│       └── types/market.ts         # TypeScript interfaces
│
├── tests/
│   ├── test_matching_engine.py     # Python LOB correctness (10 tests)
│   └── test_proto_contract.py      # Protobuf round-trip tests
│
├── docker-compose.yml              # Full stack: 8 services
├── requirements.txt                # Python dependencies
├── simulation.py                   # Python prototype simulation
├── agents.py                       # Base agent classes
├── lob.py                          # Python LOB prototype
└── .env                            # API keys (not committed)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Matching Engine | C++20, CMake, librdkafka, Protocol Buffers 3 |
| Agent Runtime | Python 3.12, Ray 2.38, asyncio, aiokafka |
| State Store | Redis 7.2 (hiredis async client) |
| Vector DB | Qdrant 1.9 (gRPC) |
| Event Bus | Apache Kafka 7.6 + Zookeeper |
| API Gateway | FastAPI, uvicorn, WebSocket |
| Frontend | Next.js 15, React 19, Tailwind CSS, TypeScript |
| Serialization | Protocol Buffers 3 (pure-Python codec + C++ native) |
| LLM Provider | Aliyun DashScope (qwen-plus / qwen-turbo) |
| Market Data | EODHD API (for SVAR empirical validation) |

---

## License

Private — internal research prototype.
