"""
src/c2gnn/api/server.py
FastAPI Alert Server for Realtime C2 Detection Pipeline.

Endpoints:
    POST /api/v1/alerts          — Receive alert from inference worker
    GET  /api/v1/alerts          — Paginated alert feed (dashboard polling)
    POST /api/v1/botnet_ips      — Register known botnet IPs (called at pipeline startup)
    GET  /api/v1/botnet_ips      — Return known botnet IP list
    GET  /api/v1/stats           — Pipeline statistics incl. TP/FP/Precision
    GET  /api/v1/health          — Health check
    WS   /api/v1/ws/alerts       — WebSocket push
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
# Pydantic Models
# ──────────────────────────────────────────────────────────────────────────────


class AlertPayload(BaseModel):
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    src_ip: str = Field(..., description="Source (bot) IP address")
    dst_ip: str = Field(..., description="Destination (C2 or peer) IP address")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="P(botnet) from GNN")
    model: str = Field(..., description="Model name: GraphSAGE | GATv2 | XGBoost")
    window_id: str | None = Field(None, description="Sliding window snapshot ID")
    reason: list[str] = Field(default_factory=list, description="Heuristic reasons")
    graph_stats: dict[str, Any] | None = Field(None, description="Graph context")
    is_known_botnet: bool | None = Field(None, description="Ground truth label (demo mode)")


class AlertResponse(BaseModel):
    alert_id: int
    received_at: str
    payload: AlertPayload


class PipelineStats(BaseModel):
    uptime_seconds: float
    total_alerts: int
    alerts_last_60s: int
    high_risk_alerts: int
    unique_src_ips: int
    unique_dst_ips: int
    top_src_ips: list[dict[str, Any]]
    model_distribution: dict[str, int]
    last_alert_at: str | None
    tp_count: int = 0
    fp_count: int = 0
    precision: float = 0.0
    known_botnet_count: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# In-memory Alert Store
# ──────────────────────────────────────────────────────────────────────────────


class AlertStore:
    def __init__(self, maxsize: int = 1000) -> None:
        self._alerts: deque[AlertResponse] = deque(maxlen=maxsize)
        self._lock = asyncio.Lock()
        self._counter = 0
        self._start_time = time.time()

        self._src_ip_counts: dict[str, int] = {}
        self._src_ip_max_score: dict[str, float] = {}
        self._dst_ip_set: set[str] = set()
        self._model_counts: dict[str, int] = {}

        # Ground truth tracking
        self._known_botnet_ips: frozenset[str] = frozenset()
        self._tp_count: int = 0
        self._fp_count: int = 0

    async def set_botnet_ips(self, ips: frozenset[str]) -> None:
        async with self._lock:
            self._known_botnet_ips = ips
            logger.info("botnet_ips_registered", count=len(ips))

    async def get_botnet_ips(self) -> list[str]:
        async with self._lock:
            return sorted(self._known_botnet_ips)

    async def add(self, payload: AlertPayload) -> AlertResponse:
        async with self._lock:
            self._counter += 1
            alert = AlertResponse(
                alert_id=self._counter,
                received_at=datetime.utcnow().isoformat() + "Z",
                payload=payload,
            )
            self._alerts.append(alert)

            src = payload.src_ip
            self._src_ip_counts[src] = self._src_ip_counts.get(src, 0) + 1
            self._src_ip_max_score[src] = max(self._src_ip_max_score.get(src, 0.0), payload.risk_score)
            self._dst_ip_set.add(payload.dst_ip)
            self._model_counts[payload.model] = self._model_counts.get(payload.model, 0) + 1

            if payload.is_known_botnet is True:
                self._tp_count += 1
            elif payload.is_known_botnet is False:
                self._fp_count += 1

            tag = ""
            if payload.is_known_botnet is True:
                tag = "TP"
            elif payload.is_known_botnet is False:
                tag = "FP"
            logger.info(
                "alert_received",
                alert_id=self._counter,
                src_ip=payload.src_ip,
                risk_score=round(payload.risk_score, 4),
                tag=tag,
            )
            return alert

    async def get_recent(
        self,
        limit: int = 50,
        offset: int = 0,
        min_score: float = 0.0,
        src_ip: str | None = None,
    ) -> list[AlertResponse]:
        async with self._lock:
            filtered = [
                a for a in self._alerts
                if a.payload.risk_score >= min_score
                and (src_ip is None or a.payload.src_ip == src_ip)
            ]
            filtered = list(reversed(filtered))
            return filtered[offset: offset + limit]

    async def get_stats(self) -> PipelineStats:
        async with self._lock:
            now = time.time()
            uptime = now - self._start_time
            total = self._counter

            cutoff = datetime.utcnow().timestamp() - 60.0
            recent_60 = sum(
                1 for a in self._alerts
                if datetime.fromisoformat(a.received_at.rstrip("Z")).timestamp() > cutoff
            )

            high_risk = sum(1 for a in self._alerts if a.payload.risk_score > 0.5)

            top_src = sorted(
                [{"ip": ip, "count": cnt, "max_score": round(self._src_ip_max_score.get(ip, 0), 4)}
                 for ip, cnt in self._src_ip_counts.items()],
                key=lambda x: x["count"],
                reverse=True,
            )[:10]

            total_tagged = self._tp_count + self._fp_count
            precision = self._tp_count / total_tagged if total_tagged > 0 else 0.0

            return PipelineStats(
                uptime_seconds=round(uptime, 1),
                total_alerts=total,
                alerts_last_60s=recent_60,
                high_risk_alerts=high_risk,
                unique_src_ips=len(self._src_ip_counts),
                unique_dst_ips=len(self._dst_ip_set),
                top_src_ips=top_src,
                model_distribution=dict(self._model_counts),
                last_alert_at=self._alerts[-1].received_at if self._alerts else None,
                tp_count=self._tp_count,
                fp_count=self._fp_count,
                precision=round(precision, 4),
                known_botnet_count=len(self._known_botnet_ips),
            )


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket Manager
# ──────────────────────────────────────────────────────────────────────────────


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
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
    description="Realtime C2 traffic detection. Powered by GraphSAGE / GATv2 on dynamic IP–IP graphs.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/api/v1/health", tags=["monitoring"])
async def health_check() -> dict[str, Any]:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z", "version": "1.0.0"}


@app.get("/health", tags=["monitoring"])
async def root_health_check() -> dict[str, Any]:
    return await health_check()


@app.post("/api/v1/botnet_ips", tags=["ground_truth"])
async def register_botnet_ips(ips: list[str]) -> dict[str, Any]:
    """Called by pipeline at startup to register known botnet IPs for TP/FP verification."""
    await store.set_botnet_ips(frozenset(ips))
    return {"status": "ok", "registered": len(ips)}


@app.get("/api/v1/botnet_ips", tags=["ground_truth"])
async def list_botnet_ips() -> list[str]:
    """Return all known botnet IPs (for dashboard Known Botnets panel)."""
    return await store.get_botnet_ips()


@app.post("/api/v1/alerts", response_model=AlertResponse, status_code=201, tags=["alerts"])
async def receive_alert(payload: AlertPayload) -> AlertResponse:
    alert = await store.add(payload)
    await ws_manager.broadcast({
        "type": "new_alert",
        "alert_id": alert.alert_id,
        "timestamp": payload.timestamp,
        "src_ip": payload.src_ip,
        "dst_ip": payload.dst_ip,
        "risk_score": payload.risk_score,
        "model": payload.model,
        "reason": payload.reason,
        "is_known_botnet": payload.is_known_botnet,
    })
    return alert


@app.get("/api/v1/alerts", response_model=list[AlertResponse], tags=["alerts"])
async def list_alerts(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    src_ip: str | None = Query(None),
) -> list[AlertResponse]:
    return await store.get_recent(limit=limit, offset=offset, min_score=min_score, src_ip=src_ip)


@app.get("/api/v1/stats", response_model=PipelineStats, tags=["monitoring"])
async def get_stats() -> PipelineStats:
    return await store.get_stats()


@app.websocket("/api/v1/ws/alerts")
async def websocket_alerts(websocket: WebSocket) -> None:
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


def run_api_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    logger.info("starting_alert_api", host=host, port=port)
    uvicorn.run("c2gnn.api.server:app", host=host, port=port, log_level="info", reload=False)


if __name__ == "__main__":
    run_api_server()
