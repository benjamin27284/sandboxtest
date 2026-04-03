# ABMS — Agent-Based Market Simulation Platform

A distributed simulation environment where **1,000 LLM-powered trading agents** interact through a **Continuous Double Auction Limit Order Book**. When news events occur, each impacted asset is simulated by heterogeneous agents trading in dedicated order books — prices emerge from agent interaction, not equations.

---

## System Architecture

```
                         ┌──────────────────────────────────┐
                         │       Next.js 15 Dashboard       │
                         │  Price chart · Order book depth  │
                         │  Shock injector · EGCIRF export  │
                         └──────────────┬───────────────────┘
                                        │ WebSocket (JSON)
                         ┌──────────────┴───────────────────┐
                         │     FastAPI  API Gateway          │
                         │  REST control · WS broadcast      │
                         │  Protobuf decode · NL analyst     │
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
     ┌─────────┴───────────────────────────────────────────────┐
     │         Three-Tier Cognitive Architecture                │
     │                                                          │
     │  Tier 1: Redis         Quantitative state (cash, VaR)   │
     │  Tier 2: Qdrant        Semantic RAG memory               │
     │  Tier 3: SLM           Episodic summaries                │
     │  Shield: QuantEngine   Bayesian · Black-Litterman · VaR  │
     └──────────────────────────────────────────────────────────┘
```

### Wire Format

All Kafka payloads between the C++ LOB engine, Python orchestrator, and API gateway use **Protocol Buffers** wire format (defined in `proto/market.proto`). The Python services use a pure-Python codec (`proto_codec.py`) that encodes/decodes protobuf without the `google.protobuf` dependency. The API gateway decodes protobuf and broadcasts JSON to WebSocket clients.

---

## Event Impact Analysis — Agent-Driven Multi-Asset Simulation

When a geopolitical or economic event occurs, the platform runs a full analysis-to-simulation pipeline. Each impacted asset is simulated by **agents trading in a limit order book**, not by Monte Carlo equations.

```
  ┌─────────────────────────────────────────────────────┐
  │                  NEWS EVENT INPUT                    │
  │  "US-Israel launch air strikes on Iran..."          │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │             LLM ANALYSIS (qwen-plus)                │
  │                                                     │
  │  Structured financial analyst prompt                 │
  │  Output: 5-15 impacted assets with:                 │
  │    - direction (up / down)                          │
  │    - magnitude (high >5% / medium 2-5% / low <2%)  │
  │    - confidence (0.0 - 1.0)                         │
  │    - causal reasoning                               │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │     PER-ASSET AGENT-DRIVEN LOB SIMULATION           │
  │                                                     │
  │  For EACH of the 15 assets independently:           │
  │                                                     │
  │    ┌─────────────────────────────────┐              │
  │    │  16 Heterogeneous Agents        │              │
  │    │  8 personas × 4 strategies      │              │
  │    │  Each calls LLM with:           │              │
  │    │    - Event context + persona    │              │
  │    │    - Current price + inventory  │              │
  │    │  → buy / sell / hold decision   │              │
  │    └──────────────┬──────────────────┘              │
  │                   │ orders                          │
  │                   ▼                                 │
  │    ┌─────────────────────────────────┐              │
  │    │  Python OrderBook (per asset)   │              │
  │    │  Price-time FIFO matching       │              │
  │    │  Fill at passive order's price  │              │
  │    │  → Executions → price updates   │              │
  │    └──────────────┬──────────────────┘              │
  │                   │                                 │
  │    Repeat for 100 ticks → OHLCV trajectory           │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │              SIMULATION OUTPUT                      │
  │                                                     │
  │  Asset          Base      Final     Trades  Change  │
  │  ─────────────────────────────────────────────────  │
  │  Crude Oil    $ 75.00   $ 82.40     156   +9.87%   │
  │  Gold         $2,350    $2,498      143   +6.30%   │
  │  S&P 500      $5,200    $4,810      189   -7.50%   │
  │  Defense      $  130    $  149      134  +14.62%   │
  │  Airlines     $ 55.00   $ 48.20     121  -12.36%   │
  │  ... (up to 15 assets)                             │
  │                                                     │
  │  Each price emerged from agent orders in a LOB,     │
  │  not from a mathematical equation.                  │
  └─────────────────────────────────────────────────────┘
```

### Why Agent-Driven, Not Monte Carlo?

The previous approach ran `asset_price = 100 + 20*sentiment + noise` fifty times per asset. That's a statistical estimator — no agents, no order book, no trading. Now:

- **16 agents per asset** with distinct personas (Momentum Trader, Contrarian Investor, Market Maker, etc.) each call the LLM to decide
- **Aggressive agents** cross the spread; **passive agents** post limit orders; **TWAP agents** slice orders
- **Market makers** continuously provide liquidity at multiple levels
- **Price discovery** happens through order flow interaction, exactly like the main simulation

---

## Data Flow Per Tick

```
  ┌──────────┐     broadcast      ┌─────────────────────────────────┐
  │          │ ──────────────────▶ │     1,000 Ray Actor Agents      │
  │          │  mid_price, spread  │                                 │
  │          │  news_headline      │  1. Load state (Redis)          │
  │ Orchest- │                     │  2. QuantEngine: Bayesian,      │
  │  rator   │                     │     Black-Litterman, VaR, DCF   │
  │          │     directives      │  3. CPT loss aversion signal    │
  │          │ ◀────────────────── │  4. Episodic flush (SLM)        │
  │          │  action, price,     │  5. RAG retrieval (Qdrant)      │
  └────┬─────┘  confidence         │  6. LLM decision → buy/sell/hold│
       │                           └─────────────────────────────────┘
       │  orders (protobuf via Kafka)
       ▼
  ┌──────────────────┐     executions (protobuf)     ┌───────────────┐
  │  C++ LOB Engine  │ ────────────────────────────▶ │  Orchestrator │
  │  854K orders/sec │                               │  routes fills  │
  │  FIFO matching   │     market_data (protobuf)    │  to actors     │
  │                  │ ────────────────────────────▶ │               │
  └──────────────────┘                               └───────┬───────┘
                                                             │
                              tick_summary (protobuf)        │
                         ┌───────────────────────────────────┘
                         ▼
                  ┌───────────────┐        JSON
                  │  API Gateway   │ ──────────────▶ Frontend
                  │  protobuf→JSON │     WebSocket
                  └───────────────┘
```

---

## Structural Causal Model (Pearl's do-calculus)

The macro environment is driven by a DAG-based SCM with structural equations:

```
  interest_rate (exogenous: 0.05 + noise)
       │
       ├──────────────────────────▶ inflation
       │                           = 0.03 - 0.4 * interest_rate + noise
       │                                │
       ▼                                ▼
  gdp_growth                     market_sentiment
  = 0.025 - 0.3*rate             = 0.5*gdp - 0.3*inflation + noise
  + 0.1*inflation + noise               │
                                  ┌──────┴──────┐
                                  ▼             ▼
                             liquidity     volatility
                             = -0.5*rate   = 0.15 - 0.2*sentiment
                             + 0.4*sent    - 0.1*liquidity + |noise|
                                  │             │
                                  └──────┬──────┘
                                         ▼
                                    asset_price
                                    = 100 + 20*sentiment
                                    + 10*liquidity - 5*volatility + noise
```

Shocks are injected via **Pearl's do-operator**: `do(variable = value)` forces a node to a fixed value, cutting all parent edges. The intervention persists for `ceil(severity * 20)` ticks and then reverts.

---

## Hierarchical Execution Framework

```
  ┌──────────────────────────────────────────┐
  │     LLM Manager Agent (decision layer)   │
  │                                          │
  │  Input:  quantitative context, news,     │
  │          portfolio state, RAG memory     │
  │  Output: {action, target_price,          │
  │           confidence, reasoning}         │
  └──────────────────┬───────────────────────┘
                     │ directive
                     ▼
  ┌──────────────────────────────────────────┐
  │     TraderSubAgent (execution layer)     │
  │     Pure rule-based — no LLM calls       │
  │                                          │
  │  ┌────────────┐  ┌──────────┐            │
  │  │ aggressive  │  │ passive  │            │
  │  │ +slippage   │  │ at limit │            │
  │  └────────────┘  └──────────┘            │
  │  ┌────────────┐  ┌──────────┐            │
  │  │    TWAP    │  │   DDQL   │            │
  │  │ 3 tranches │  │ learned  │            │
  │  └────────────┘  └──────────┘            │
  └──────────────────┬───────────────────────┘
                     │ order (protobuf)
                     ▼
               Kafka → C++ LOB
```

---

## Agent Heterogeneity

1,000 agents are assigned from **8 persona archetypes** with different Bayesian sophistication levels:

| Persona | Sophistication | Behavior |
|---------|---------------|----------|
| Fundamental Value Fund Manager | 0.9 | Slightly underweights signals (conservative) |
| Macro Hedge Fund Strategist | 1.0 | Fully rational Bayesian |
| Momentum Trader | 1.1 | Overweights signals (trend-chasing) |
| Contrarian Investor | 0.85 | Goes against consensus |
| Risk-Parity Portfolio Manager | 1.0 | Balanced, diversification-focused |
| Event-Driven Arbitrageur | 1.1 | Overweights event signals aggressively |
| Market Maker | 1.0 | Spread-focused, liquidity provider |
| Behavioral Finance Specialist | 0.95 | Behavioral bias aware |

Each agent also has one of 4 execution strategies (aggressive, passive, TWAP, DDQL) assigned round-robin. 25% of agents use Double Deep Q-Learning for adaptive execution.

---

## Deterministic Math Shield

The LLM **never computes numerical results**. All quantitative analysis is deterministic and injected into the prompt as `[SOURCE:TAG]`-wrapped facts:

| Module | Method | Purpose |
|--------|--------|---------|
| Bayesian Update | Conjugate Gaussian | Posterior belief from prior + market signal |
| Black-Litterman | Theil mixed estimation | Optimal position size |
| Parametric VaR | z-score x vol x sqrt(T) | 95% confidence risk measure |
| DCF Fair Value | Multi-stage + Gordon Growth | Fundamental valuation |
| CPT Signal | Kahneman-Tversky (lambda=2.25) | Loss aversion signal in [-1, 1] |

---

## MASS Framework (Multi-Agent System Signal)

Quantifies emergent consensus and disagreement across all 1,000 agents per tick:

```
Signal(s,j) = alpha * m_s(j) - (1 - alpha) * sigma_s(j)

where:
  m_s     = mean directional score (buy=+conf, sell=-conf, hold=0)
  sigma_s = stdev of scores (disagreement)
  alpha   = 0.6 (consensus weight)
```

---

## Project Structure

```
.
├── proto/
│   └── market.proto                # Shared protobuf schema (source of truth)
│
├── services/
│   ├── lob-engine/                 # C++20 matching engine (854K orders/sec)
│   │   ├── include/
│   │   │   ├── order.h             #   Order + Execution structs
│   │   │   └── orderbook.h         #   OrderBook (std::map + deque)
│   │   ├── src/
│   │   │   ├── orderbook.cpp       #   Price-time FIFO matching
│   │   │   └── main.cpp            #   Kafka consumer/producer loop
│   │   ├── tests/
│   │   │   └── bench_orderbook.cpp #   Correctness + 2M-order benchmark
│   │   ├── CMakeLists.txt
│   │   └── Dockerfile
│   │
│   ├── agent-orchestrator/         # Python / Ray (1,000 agents)
│   │   ├── actors/
│   │   │   └── base_actor.py       #   TradingAgentActor (8 personas, 3-tier)
│   │   ├── memory/
│   │   │   ├── state_store.py      #   Tier 1: Redis state
│   │   │   ├── semantic_memory.py  #   Tier 2: Qdrant RAG
│   │   │   └── episodic_buffer.py  #   Tier 3: SLM episodic buffer
│   │   ├── math_engine/
│   │   │   └── quant_models.py     #   Bayesian, BL, VaR, DCF, CPT
│   │   ├── config/
│   │   │   └── settings.py         #   Environment-based configuration
│   │   ├── orchestrator.py         #   Main tick loop + Kafka + MASS
│   │   ├── matching_engine.py      #   Python LOB for multi-asset event sims
│   │   ├── proto_codec.py          #   Pure-Python protobuf wire codec
│   │   ├── event_impact_analyzer.py#   LLM analysis + agent-driven per-asset sim
│   │   ├── causal_engine.py        #   Pearl's SCM + do-calculus + EGCIRF
│   │   ├── ddql_agent.py           #   Double DQN execution agent
│   │   ├── svar_validation.py      #   Guerini-Moneta SVAR validation
│   │   ├── run_event_analysis.py   #   Standalone event analysis CLI
│   │   └── Dockerfile
│   │
│   └── api-gateway/                # FastAPI + WebSocket bridge
│       ├── src/
│       │   └── main.py             #   REST + WS + protobuf decode
│       └── Dockerfile
│
├── frontend/                       # Next.js 15 + React 19 + Tailwind CSS
│   ├── src/
│   │   ├── types/market.ts         #   TypeScript interfaces (protobuf-aligned)
│   │   ├── hooks/useMarketSocket.ts#   WebSocket hook + auto-reconnect
│   │   ├── components/
│   │   │   ├── charts/PriceChart.tsx
│   │   │   └── controls/ShockInjector.tsx
│   │   └── app/dashboard/page.tsx  #   Main dashboard
│   ├── package.json
│   └── Dockerfile
│
├── tests/
│   ├── test_matching_engine.py     #   Python LOB correctness (10 tests)
│   └── test_proto_contract.py      #   Protobuf round-trip + frontend shape (10 tests)
│
├── docker-compose.yml              #   Full stack: 7 services + 3 data stores
├── lob.py                          #   Python LOB prototype (heapq-based)
├── agents.py                       #   Base agent classes prototype
├── llm_agent.py                    #   LLM agent prototype (DashScope API)
└── simulation.py                   #   50-tick prototype simulation
```

---

## Quick Start

### Full Distributed Stack (Docker)

```bash
cp .env.example .env   # Add DASHSCOPE_API_KEY and EODHD_API_KEY
docker compose up -d
open http://localhost:3000/dashboard
```

### Python Prototype Only

```bash
pip install -r requirements.txt
python simulation.py --ticks 50
```

### C++ Benchmark

```bash
cd services/lob-engine
cmake -B out -DCMAKE_BUILD_TYPE=Release
cmake --build out --parallel
./out/lob_bench
```

Expected output:
```
=== OrderBook Correctness Tests ===
  [PASS] test_basic_match
  [PASS] test_partial_fill_multi_level
  [PASS] test_market_order
  [PASS] test_cancel
  [PASS] test_get_mid_price

=== OrderBook Throughput Benchmark ===
  [BENCH] 2000000 orders in 2342.1 ms = 853952 orders/sec
```

### Event Impact Analysis

```bash
cd services/agent-orchestrator
python run_event_analysis.py            # Live (requires DASHSCOPE_API_KEY)
python run_event_analysis.py --offline  # Cached response (no API needed)
```

### Run Tests

```bash
python tests/test_matching_engine.py
python tests/test_proto_contract.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + WS connection count |
| POST | `/api/simulation/control` | Start / stop / pause simulation |
| POST | `/api/shocks/inject` | Inject exogenous shock |
| POST | `/api/reports/svar` | Guerini-Moneta SVAR validation |
| GET | `/api/reports/egcirf` | Counterfactual impulse response report |
| POST | `/api/nl/query` | Natural language analyst query |
| WS | `/ws/market` | Real-time market telemetry stream |

---

## Kafka Topics

All topics use protobuf serialization (defined in `proto/market.proto`):

| Topic | Partitions | Direction | Message Type |
|-------|-----------|-----------|--------------|
| `orders_submit` | 8 | Orchestrator → LOB | Order |
| `orders_cancel` | 4 | Orchestrator → LOB | CancelOrder |
| `executions` | 8 | LOB → Orchestrator, Gateway | Execution |
| `market_data` | 4 | LOB → Orchestrator, Gateway | MarketSnapshot |
| `tick_summary` | 1 | Orchestrator → Gateway | TickSummary |
| `exogenous_shocks` | 1 | Gateway → Orchestrator | JSON (ExogenousShock) |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_BROKERS` | `localhost:9092` | Kafka bootstrap servers |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `QDRANT_HOST` | `localhost` | Qdrant vector DB host |
| `DASHSCOPE_API_KEY` | — | Aliyun DashScope API key |
| `DASHSCOPE_BASE_URL` | `https://dashscope-us.aliyuncs.com/compatible-mode/v1` | LLM endpoint |
| `EODHD_API_KEY` | — | EODHD market data API key |
| `NUM_AGENTS` | `1000` | Number of trading agents |
| `PRIMARY_LLM_MODEL` | `qwen-plus` | Main LLM model |
| `SLM_MODEL` | `qwen-turbo` | Small model for episodic summaries |
| `RAY_NUM_CPUS` | `8` | Ray cluster CPU allocation |
| `LLM_TIMEOUT_SECONDS` | `15` | LLM call timeout |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Matching Engine | C++20, CMake, librdkafka, Protocol Buffers 3 |
| Agent Runtime | Python 3.12, Ray 2.38, asyncio, aiokafka |
| State Store | Redis 7.2 (hiredis) |
| Vector DB | Qdrant 1.9 (gRPC) |
| Event Bus | Apache Kafka (Confluent 7.6) + Zookeeper |
| API Gateway | FastAPI, uvicorn, WebSocket |
| Frontend | Next.js 15, React 19, Tailwind CSS, TypeScript |
| Serialization | Protocol Buffers 3 (pure-Python codec + C++ protobuf) |
| LLM Provider | Aliyun DashScope (qwen-plus / qwen-turbo) |

---

## License

Private — internal research prototype.
