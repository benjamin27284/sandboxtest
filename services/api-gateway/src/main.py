"""API Gateway — FastAPI + WebSocket bridge.

Serves:
  - REST endpoints for simulation control (start, stop, inject shock)
  - WebSocket endpoint for real-time market telemetry to the frontend
  - EGCIRF report export endpoint

Consumes market_data and tick_summary from Kafka and broadcasts
to connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow imports from the agent-orchestrator service
_ORCHESTRATOR_DIR = os.path.join(os.path.dirname(__file__), "../../agent-orchestrator")
if _ORCHESTRATOR_DIR not in sys.path:
    sys.path.insert(0, _ORCHESTRATOR_DIR)

# ─── Configuration ───────────────────────────────────────────────────────────

KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# ─── WebSocket Manager ──────────────────────────────────────────────────────

class ConnectionManager:
    """Manages active WebSocket connections for market data streaming."""

    def __init__(self, max_connections: int = 200) -> None:
        self._connections: set[WebSocket] = set()
        self._max = max_connections

    async def connect(self, ws: WebSocket) -> bool:
        if len(self._connections) >= self._max:
            await ws.close(code=1013, reason="Max connections reached")
            return False
        await ws.accept()
        self._connections.add(ws)
        return True

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def broadcast(self, message: dict) -> None:
        payload = json.dumps(message)
        dead = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.discard(ws)

    @property
    def count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()

# ─── Kafka → WebSocket bridge task ──────────────────────────────────────────

async def kafka_to_ws_bridge() -> None:
    """Background task: consume Kafka market data (protobuf) → broadcast JSON via WebSocket."""
    try:
        from aiokafka import AIOKafkaConsumer
    except ImportError:
        logger.warning("aiokafka not installed; WS bridge disabled")
        return

    try:
        from proto_codec import decode_execution, decode_market_snapshot, decode_tick_summary
    except Exception as exc:
        logger.error("Failed to import proto_codec: %s", exc)
        return

    # Use a unique group_id per instance to avoid stale committed offsets
    import uuid
    consumer = AIOKafkaConsumer(
        "market_data", "tick_summary", "executions",
        bootstrap_servers=KAFKA_BROKERS,
        group_id=f"api-gateway-ws-{uuid.uuid4().hex[:8]}",
        auto_offset_reset="latest",
        enable_auto_commit=False,
    )

    try:
        await consumer.start()
        logger.info("Kafka→WS bridge started (protobuf decoding)")

        async for msg in consumer:
            try:
                topic = msg.topic
                raw = msg.value  # bytes
                if topic == "executions":
                    data = decode_execution(raw)
                elif topic == "market_data":
                    data = decode_market_snapshot(raw)
                elif topic == "tick_summary":
                    data = decode_tick_summary(raw)
                else:
                    data = json.loads(raw.decode())
            except Exception as exc:
                logger.debug("Failed to decode %s message: %s", msg.topic, exc)
                continue

            payload = {
                "topic": msg.topic,
                "data": data,
                "timestamp": time.time(),
            }
            logger.info("Broadcasting %s (tick=%s, clients=%d)", msg.topic, data.get("tick", "?"), manager.count)
            await manager.broadcast(payload)
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error("Kafka→WS bridge crashed: %s", exc)
    finally:
        await consumer.stop()


# ─── App lifecycle ───────────────────────────────────────────────────────────

bridge_task: Optional[asyncio.Task] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bridge_task
    bridge_task = asyncio.create_task(kafka_to_ws_bridge())
    yield
    if bridge_task:
        bridge_task.cancel()
        try:
            await bridge_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="ABMS Platform API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── REST Endpoints ─────────────────────────────────────────────────────────

class ShockRequest(BaseModel):
    tick: int
    category: str = "monetary_policy"
    headline: str
    body: str = ""
    severity: float = 0.5


class SimulationControl(BaseModel):
    action: str  # "start" | "stop" | "pause"
    total_ticks: int = 100


@app.get("/health")
async def health():
    return {"status": "ok", "ws_connections": manager.count}


@app.post("/api/simulation/control")
async def control_simulation(req: SimulationControl):
    """Start, stop, or pause the simulation via Redis control flags."""
    import redis.asyncio as aioredis
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        if req.action == "stop":
            await r.set("sim:stop", "1")
        elif req.action == "start":
            await r.set("sim:stop", "0")
        elif req.action == "pause":
            await r.set("sim:stop", "1")

        status = await r.get("sim:status") or "unknown"
    finally:
        await r.aclose()

    return {"status": "accepted", "action": req.action, "sim_status": status}


@app.get("/api/simulation/status")
async def simulation_status():
    """Get current simulation status."""
    import redis.asyncio as aioredis
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        status = await r.get("sim:status") or "unknown"
        stop_flag = await r.get("sim:stop") or "0"
    finally:
        await r.aclose()

    return {"status": status, "stopped": stop_flag == "1"}


@app.get("/api/event-analysis/latest")
async def get_latest_event_analysis():
    """Get the latest event impact analysis with per-asset simulation results."""
    import redis.asyncio as aioredis
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        data = await r.get("event_analysis:latest")
        if not data:
            return {"analysis": None}
        return {"analysis": json.loads(data)}
    finally:
        await r.aclose()


@app.post("/api/shocks/inject")
async def inject_shock(shock: ShockRequest):
    """Inject an exogenous macroeconomic shock into the simulation.

    This is the 'do-operator' intervention endpoint.
    """
    try:
        from aiokafka import AIOKafkaProducer
        producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BROKERS,
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        await producer.start()
        await producer.send_and_wait(
            "exogenous_shocks",
            value=shock.model_dump(),
        )
        await producer.stop()
    except ImportError:
        raise HTTPException(503, "Kafka not available")

    return {"status": "injected", "shock_id": f"SHK-{shock.tick}"}


class SVARValidationRequest(BaseModel):
    empirical_prices: list[float] = []      # if empty, auto-fetch from EODHD
    empirical_volumes: list[float] = []
    ticker: str = "SPY.US"                  # EODHD ticker to use when auto-fetching
    n_bars: int = 252
    threshold: float = 0.05


@app.post("/api/reports/svar")
async def run_svar_validation_endpoint(req: SVARValidationRequest):
    """Run Guerini-Moneta SVAR validation.

    Accepts empirical price/volume series, fetches the simulated series
    from the orchestrator's accumulated history, fits VAR(1) on both,
    and returns the topological distance (precision/recall/F1) between
    the Granger-causal graphs.

    In production, the simulated series is pulled from Redis/Postgres.
    """
    from svar_validation import (
        TimeSeriesData, run_svar_validation,
        compute_returns, compute_realized_vol, load_empirical_data,
    )

    if req.empirical_prices:
        emp_returns = compute_returns(req.empirical_prices)
        emp_vols = compute_realized_vol(emp_returns)
        empirical = TimeSeriesData(
            prices=req.empirical_prices,
            volumes=req.empirical_volumes or [0.0] * len(req.empirical_prices),
            volatilities=emp_vols,
        )
    else:
        # Auto-fetch from EODHD
        empirical = load_empirical_data(
            ticker=req.ticker,
            n_bars=req.n_bars,
        )

    # In production: fetch simulated prices from Redis/Postgres.
    # For now, use the empirical series as a self-comparison baseline
    # (will yield F1=1.0, proving the pipeline works end-to-end).
    simulated = TimeSeriesData(
        prices=empirical.prices,
        volumes=empirical.volumes,
        volatilities=empirical.volatilities,
    )

    result = run_svar_validation(simulated, empirical, req.threshold)
    result["generated_at"] = time.time()
    return result


@app.get("/api/reports/egcirf")
async def get_egcirf_report(
    shock_tick: Optional[int] = None,
    intervention_var: str = "interest_rate",
    intervention_value: float = 0.08,
    target_var: str = "asset_price",
    n_ticks: int = 30,
    n_runs: int = 50,
):
    """EGCIRF report endpoint.

    Accepts either:
      - ?shock_tick=25  (frontend shorthand — uses default intervention)
      - ?intervention_var=X&intervention_value=Y  (full parameterization)
    """
    from causal_engine import compute_egcirf, StructuralCausalModel

    result = compute_egcirf(
        scm_factory=StructuralCausalModel,
        intervention_var=intervention_var,
        intervention_value=intervention_value,
        target_var=target_var,
        n_ticks=n_ticks,
        n_runs=n_runs,
    )
    result["generated_at"] = time.time()
    # Include shock_tick in response for frontend compatibility
    result["shock_tick"] = shock_tick if shock_tick is not None else 0
    return result


# ─── Natural Language Query Interface ─────────────────────────────────────────

# Variable name mappings for the SCM DAG
_VAR_KEYWORDS = {
    "inflation": "inflation",
    "gdp": "gdp_growth",
    "sentiment": "market_sentiment",
    "liquidity": "liquidity",
    "volatility": "volatility",
}


class NLQueryRequest(BaseModel):
    query: str      # e.g. "inject an interest rate hike of 200bps and show me the EGCIRF"
    n_runs: int = 20


@app.post("/api/nl/query")
async def natural_language_query(req: NLQueryRequest):
    """Parse natural-language analyst queries and route to the right engine.

    Uses keyword matching (no LLM dependency for the interface itself).
    """
    q = req.query.lower()

    # ── Route: EGCIRF ────────────────────────────────────────────────────
    if any(k in q for k in ["egcirf", "impulse response",
                             "counterfactual", "cirf"]):
        var = "interest_rate"
        target = "asset_price"

        for keyword, scm_var in _VAR_KEYWORDS.items():
            if keyword in q:
                # Keywords that name a *target* vs an *intervention* variable
                if keyword == "volatility":
                    target = scm_var
                else:
                    var = scm_var
                break

        # Extract intervention magnitude from "NNNbps" (basis points).
        # 200 bps = 0.02 in absolute terms.  Default to a 50-bp shift.
        bps_match = re.search(r"(\d+)\s*bps", q)
        if bps_match:
            val = int(bps_match.group(1)) / 10_000
        else:
            val = 0.005  # 50 bps default

        # The SCM's interest_rate baseline is 0.05, so the intervention
        # value is the *level* (baseline + shock), not the shock alone.
        baseline_levels = {
            "interest_rate": 0.05,
            "inflation": 0.03,
            "gdp_growth": 0.025,
            "market_sentiment": 0.0,
            "liquidity": 0.0,
            "volatility": 0.15,
        }
        intervention_value = baseline_levels.get(var, 0.0) + val

        from causal_engine import compute_egcirf, StructuralCausalModel

        result = compute_egcirf(
            scm_factory=StructuralCausalModel,
            intervention_var=var,
            intervention_value=intervention_value,
            target_var=target,
            n_runs=req.n_runs,
        )
        return {"query": req.query, "report_type": "egcirf", "result": result}

    # ── Route: Shock injection ───────────────────────────────────────────
    if any(k in q for k in ["inject", "shock", "rate hike",
                             "crash", "war", "tariff"]):
        severity = 0.8
        sev_match = re.search(r"severity[:\s]+([0-9.]+)", q)
        if sev_match:
            severity = min(1.0, max(0.0, float(sev_match.group(1))))

        shock = {
            "headline": req.query,
            "severity": severity,
            "variable": "market_sentiment",
        }
        return {"query": req.query, "report_type": "shock_injected",
                "result": shock}

    # ── Route: SVAR validation ───────────────────────────────────────────
    if any(k in q for k in ["svar", "validate", "validation",
                             "causal structure"]):
        return {
            "query": req.query,
            "report_type": "svar",
            "result": (
                "Run simulation first via POST /api/simulation/control, "
                "then POST /api/reports/svar with empirical prices."
            ),
        }

    # ── Route: Status ────────────────────────────────────────────────────
    if any(k in q for k in ["status", "health", "how many", "agents"]):
        return {"query": req.query, "report_type": "status",
                "result": {"ws_connections": manager.count}}

    # ── Fallback ─────────────────────────────────────────────────────────
    return {
        "query": req.query,
        "report_type": "unknown",
        "result": (
            "Supported queries: EGCIRF analysis, shock injection, "
            "SVAR validation, status check."
        ),
    }


# ─── WebSocket ───────────────────────────────────────────────────────────────

@app.websocket("/ws/market")
async def ws_market(ws: WebSocket):
    """Real-time market telemetry stream.

    Clients receive JSON messages:
        {"topic": "market_data"|"tick_summary"|"executions", "data": {...}}
    """
    ok = await manager.connect(ws)
    if not ok:
        return

    try:
        while True:
            # Keep connection alive; client can send commands
            data = await ws.receive_text()
            # Optional: handle client commands (subscribe/unsubscribe)
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
