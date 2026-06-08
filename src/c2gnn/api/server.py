"""
src/c2gnn/api/server.py
FastAPI Alert Server for Realtime C2 Detection Pipeline.

Receives Alert JSON from Thread 3, stores in ring buffer, exposes REST API
for the Streamlit dashboard and external SIEM integration.

Endpoints:
    POST /api/v1/alerts          — Receive alert from inference worker
    GET  /api/v1/alerts          — Paginated alert feed (dashboard polling)
    GET  /api/v1/stats           — Pipeline statistics
    GET  /api/v1/health          — Health check
    WS   /api/v1/ws/alerts       — WebSocket push (optional real-time)
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from datetime import datetime
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Models (request / response schemas)
# ──────────────────────────────────────────────────────────────────────────────


class AlertPayload(BaseModel):
    """Alert posted by the InferenceWorker (Thread 3)."""

    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    src_ip: str = Field(..., description="Source (bot) IP address")
    dst_ip: str = Field(..., description="Destination (C2 or peer) IP address")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="P(botnet) from GNN")
    model: str = Field(..., description="Model name: GraphSAGE | GATv2 | XGBoost")
    window_id: str | None = Field(None, description="Sliding window snapshot ID")
    reason: list[str] = Field(default_factory=list, description="Heuristic reasons")
    graph_stats: dict[str, Any] | None = Field(
        None, description="Graph context: node count, edge count, etc."
    )


class AlertResponse(BaseModel):
    """Alert stored in ring buffer, with server-assigned ID."""

    alert_id: int
    received_at: str  # Server timestamp
    payload: AlertPayload


class PipelineStats(BaseModel):
    """Aggregated statistics from the pipeline."""

    uptime_seconds: float
    total_alerts: int
    alerts_last_60s: int
    high_risk_alerts: int  # risk_score > 0.9
    unique_src_ips: int
    unique_dst_ips: int
    top_src_ips: list[dict[str, Any]]  # [{ip, count, max_score}]
    model_distribution: dict[str, int]  # {model_name: count}
    last_alert_at: str | None


# ──────────────────────────────────────────────────────────────────────────────
# In-memory Alert Store
# ──────────────────────────────────────────────────────────────────────────────


class AlertStore:
    """Thread-safe in-memory ring buffer for recent alerts.

    Uses asyncio.Lock — suitable for async FastAPI handlers.
    For production, swap with Redis Streams or TimescaleDB.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._alerts: deque[AlertResponse] = deque(maxlen=maxsize)
        self._lock = asyncio.Lock()
        self._counter = 0
        self._start_time = time.time()

        # Stats tracking
        self._src_ip_counts: dict[str, int] = {}
        self._src_ip_max_score: dict[str, float] = {}
        self._dst_ip_set: set[str] = set()
        self._model_counts: dict[str, int] = {}

    async def add(self, payload: AlertPayload) -> AlertResponse:
        """Add a new alert; returns the stored AlertResponse."""
        async with self._lock:
            self._counter += 1
            alert = AlertResponse(
                alert_id=self._counter,
                received_at=datetime.utcnow().isoformat() + "Z",
                payload=payload,
            )
            self._alerts.append(alert)

            # Update stats
            src = payload.src_ip
            self._src_ip_counts[src] = self._src_ip_counts.get(src, 0) + 1
            self._src_ip_max_score[src] = max(
                self._src_ip_max_score.get(src, 0.0), payload.risk_score
            )
            self._dst_ip_set.add(payload.dst_ip)
            self._model_counts[payload.model] = self._model_counts.get(payload.model, 0) + 1

            logger.info(
                "alert_received",
                alert_id=self._counter,
                src_ip=payload.src_ip,
                dst_ip=payload.dst_ip,
                risk_score=round(payload.risk_score, 4),
                model=payload.model,
            )
            return alert

    async def get_recent(
        self,
        limit: int = 50,
        offset: int = 0,
        min_score: float = 0.0,
        src_ip: str | None = None,
    ) -> list[AlertResponse]:
        """Return recent alerts with optional filters."""
        async with self._lock:
            filtered = [
                a
                for a in self._alerts
                if a.payload.risk_score >= min_score
                and (src_ip is None or a.payload.src_ip == src_ip)
            ]
            # Newest first
            filtered = list(reversed(filtered))
            return filtered[offset : offset + limit]

    async def get_stats(self) -> PipelineStats:
        """Compute aggregated statistics."""
        async with self._lock:
            now = time.time()
            uptime = now - self._start_time
            total = self._counter

            # Alerts in last 60s
            cutoff = datetime.utcnow().timestamp() - 60.0
            recent_60 = sum(
                1
                for a in self._alerts
                if datetime.fromisoformat(a.received_at.rstrip("Z")).timestamp() > cutoff
            )

            high_risk = sum(1 for a in self._alerts if a.payload.risk_score > 0.9)

            top_src = sorted(
                [
                    {
                        "ip": ip,
                        "count": cnt,
                        "max_score": round(self._src_ip_max_score.get(ip, 0), 4),
                    }
                    for ip, cnt in self._src_ip_counts.items()
                ],
                key=lambda x: x["count"],
                reverse=True,
            )[:10]

            last_alert_at = self._alerts[-1].received_at if self._alerts else None

            return PipelineStats(
                uptime_seconds=round(uptime, 1),
                total_alerts=total,
                alerts_last_60s=recent_60,
                high_risk_alerts=high_risk,
                unique_src_ips=len(self._src_ip_counts),
                unique_dst_ips=len(self._dst_ip_set),
                top_src_ips=top_src,
                model_distribution=dict(self._model_counts),
                last_alert_at=last_alert_at,
            )


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket Connection Manager
# ──────────────────────────────────────────────────────────────────────────────


class ConnectionManager:
    """Manages active WebSocket connections for push-based dashboard updates."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("ws_client_connected", total=len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)
        logger.info("ws_client_disconnected", total=len(self.active_connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast alert to all connected WebSocket clients."""
        dead: list[WebSocket] = []
        for ws in self.active_connections:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active_connections.remove(ws)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────────────────────────────

store = AlertStore(maxsize=1000)
ws_manager = ConnectionManager()

app = FastAPI(
    title="C2 GNN Detection — Alert API",
    description=(
        "Realtime C2 traffic detection alert feed. "
        "Powered by GraphSAGE / GATv2 on dynamic IP–IP graphs."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/api/v1/health", tags=["monitoring"])
async def health_check() -> dict[str, Any]:
    """Liveness probe."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0",
    }


@app.post(
    "/api/v1/alerts",
    response_model=AlertResponse,
    status_code=201,
    tags=["alerts"],
)
async def receive_alert(payload: AlertPayload) -> AlertResponse:
    """Receive an alert from the InferenceWorker (Thread 3).

    This endpoint is called by the pipeline; not by human users.
    """
    alert = await store.add(payload)

    # Push to WebSocket subscribers
    await ws_manager.broadcast(
        {
            "type": "new_alert",
            "alert_id": alert.alert_id,
            "timestamp": payload.timestamp,
            "src_ip": payload.src_ip,
            "dst_ip": payload.dst_ip,
            "risk_score": payload.risk_score,
            "model": payload.model,
            "reason": payload.reason,
        }
    )

    return alert


@app.get(
    "/api/v1/alerts",
    response_model=list[AlertResponse],
    tags=["alerts"],
)
async def list_alerts(
    limit: int = Query(50, ge=1, le=500, description="Max alerts to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    min_score: float = Query(0.0, ge=0.0, le=1.0, description="Minimum risk score"),
    src_ip: str | None = Query(None, description="Filter by source IP"),
) -> list[AlertResponse]:
    """Get recent alerts — polled by the Streamlit dashboard."""
    return await store.get_recent(limit=limit, offset=offset, min_score=min_score, src_ip=src_ip)


@app.get(
    "/api/v1/stats",
    response_model=PipelineStats,
    tags=["monitoring"],
)
async def get_stats() -> PipelineStats:
    """Aggregated pipeline statistics for the dashboard header."""
    return await store.get_stats()


@app.websocket("/api/v1/ws/alerts")
async def websocket_alerts(websocket: WebSocket) -> None:
    """WebSocket endpoint — push new alerts to connected dashboard clients."""
    await ws_manager.connect(websocket)
    try:
        # Keep connection alive — client sends pings
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────


def run_api_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the FastAPI alert server.

    Called by the RealtimePipeline orchestrator or standalone via:
        python -m c2gnn.api.server
    """
    logger.info("starting_alert_api", host=host, port=port)
    uvicorn.run(
        "c2gnn.api.server:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    run_api_server()
