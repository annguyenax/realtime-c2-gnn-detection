"""
realtime/pipeline.py — 3-Thread Realtime C2 Detection Pipeline

Architecture:
  Thread 1 — FlowBuilderWorker:
    CTU-13 file → replay by timestamp → flow_queue (Queue[FlowRecord])

  Thread 2 — GraphUpdateWorker:
    flow_queue → update SlidingWindowGraph → inference_queue (Queue[PyG Data])

  Thread 3 — InferenceWorker:
    inference_queue → GNN forward pass → Alert JSON → callback/API

Author: Member 1 (Threads 1-2) + Member 2 (Thread 3 + orchestration)
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog
import torch
import torch.nn as nn

from c2gnn.data.flow_builder import FlowBuilderWorker
from c2gnn.graph.dynamic_graph import SlidingWindowGraph

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Alert data model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Alert:
    """
    Structured alert emitted when a node exceeds the detection threshold.
    JSON-serializable for API output and SIEM integration.
    """

    timestamp: float
    src_ip: str
    dst_ip: str | None
    risk_score: float
    model: str
    reasons: list[str]
    graph_version: int
    inference_latency_ms: float
    graph_nodes: int
    graph_edges: int

    def to_json(self) -> str:
        d = asdict(self)
        d["dst_ip"] = d["dst_ip"] or "unknown"
        d["timestamp_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp))
        return json.dumps(d, indent=2)

    def __str__(self) -> str:
        return (
            f"🚨 ALERT  ip={self.src_ip}  "
            f"score={self.risk_score:.3f}  "
            f"model={self.model}  "
            f"latency={self.inference_latency_ms:.1f}ms  "
            f"reasons={self.reasons}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Thread 2 — Graph Update Worker
# ─────────────────────────────────────────────────────────────────────────────


class GraphUpdateWorker(threading.Thread):
    """
    Thread 2: Consumes FlowRecords, maintains SlidingWindowGraph,
    periodically pushes PyG Data snapshots to inference_queue.

    Snapshot strategy:
      - Push snapshot every `snapshot_interval` seconds (time-based)
      - Also push if `snapshot_on_n_flows` flows have accumulated (count-based)
      - This avoids both high-frequency snapshots and stale detection windows
    """

    def __init__(
        self,
        flow_queue: queue.Queue,
        inference_queue: queue.Queue,
        window_size: float = 60.0,
        edge_ttl: float = 120.0,
        snapshot_interval: float = 5.0,
        snapshot_on_n_flows: int = 500,
        stop_event: threading.Event | None = None,
    ):
        super().__init__(name="GraphUpdater", daemon=True)
        self.flow_queue = flow_queue
        self.inference_queue = inference_queue
        self.graph = SlidingWindowGraph(
            window_size=window_size,
            edge_ttl=edge_ttl,
        )
        self.snapshot_interval = snapshot_interval
        self.snapshot_on_n_flows = snapshot_on_n_flows
        self.stop_event = stop_event or threading.Event()

        self._flows_since_snapshot = 0
        self._last_snapshot_wall = 0.0
        self._snapshots_pushed = 0
        self._flows_processed = 0

    def run(self) -> None:
        log = logger.bind(thread="GraphUpdater")
        log.info("Starting", window_size=self.graph.window_size)

        while not self.stop_event.is_set():
            try:
                flow = self.flow_queue.get(timeout=1.0)
            except queue.Empty:
                # Even if no new flows, push a snapshot at interval
                self._maybe_push_snapshot(log)
                continue

            if flow is None:  # Sentinel from FlowBuilder
                log.info("Sentinel received, flushing final snapshot")
                self._push_snapshot(log, reason="final")
                self.inference_queue.put(None)  # Forward sentinel
                break

            # Incremental update
            t_update_start = time.perf_counter()
            self.graph.update(flow)
            update_ms = (time.perf_counter() - t_update_start) * 1000

            self._flows_processed += 1
            self._flows_since_snapshot += 1

            if update_ms > 100:
                log.warning("Slow graph update", update_ms=round(update_ms, 1))

            # Decide whether to push snapshot
            time_trigger = time.time() - self._last_snapshot_wall >= self.snapshot_interval
            count_trigger = self._flows_since_snapshot >= self.snapshot_on_n_flows

            if time_trigger or count_trigger:
                reason = "time" if time_trigger else "count"
                self._push_snapshot(log, reason=reason)

        log.info(
            "Done",
            flows_processed=self._flows_processed,
            snapshots_pushed=self._snapshots_pushed,
        )

    def _maybe_push_snapshot(self, log) -> None:
        if time.time() - self._last_snapshot_wall >= self.snapshot_interval:
            self._push_snapshot(log, reason="idle")

    def _push_snapshot(self, log, reason: str = "scheduled") -> None:
        """Convert current graph to PyG Data and push to inference_queue."""
        if self.graph.num_nodes == 0:
            return

        t0 = time.perf_counter()
        pyg_data = self.graph.to_pyg_data(include_ground_truth=False)
        convert_ms = (time.perf_counter() - t0) * 1000

        if pyg_data is not None:
            try:
                self.inference_queue.put(pyg_data, timeout=2.0)
                self._snapshots_pushed += 1
                self._flows_since_snapshot = 0
                self._last_snapshot_wall = time.time()

                log.debug(
                    "Snapshot pushed",
                    reason=reason,
                    nodes=pyg_data.num_nodes,
                    edges=pyg_data.num_edges,
                    convert_ms=round(convert_ms, 1),
                    snapshot_n=self._snapshots_pushed,
                )
            except queue.Full:
                log.warning("Inference queue full, dropping snapshot")


# ─────────────────────────────────────────────────────────────────────────────
# Thread 3 — Inference Worker
# ─────────────────────────────────────────────────────────────────────────────


class InferenceWorker(threading.Thread):
    """
    Thread 3: Consumes PyG Data snapshots, runs GNN, emits alerts.

    Alert deduplication:
      - Same IP not alerted more than once per `dedup_window_s` seconds
      - Prevents alert flood when a bot is continuously active

    Args:
        inference_queue: Input queue from GraphUpdateWorker
        model: Trained PyTorch model (GraphSAGE or GATv2)
        threshold: Probability threshold for alert (default 0.7)
        alert_callback: Function called with Alert object
        dedup_window_s: Seconds before same IP can be re-alerted
    """

    def __init__(
        self,
        inference_queue: queue.Queue,
        model: nn.Module,
        threshold: float = 0.7,
        alert_callback: Callable[[Alert], None] | None = None,
        dedup_window_s: float = 30.0,
        stop_event: threading.Event | None = None,
    ):
        super().__init__(name="InferenceWorker", daemon=True)
        self.inference_queue = inference_queue
        self.threshold = threshold
        self.alert_callback = alert_callback or self._default_handler
        self.dedup_window_s = dedup_window_s
        self.stop_event = stop_event or threading.Event()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        # Dedup: ip → last_alert_timestamp
        self._last_alert: dict[str, float] = {}
        self._inference_count = 0
        self._alert_count = 0

        # Latency tracking
        self._latencies_ms: list[float] = []

    def run(self) -> None:
        log = logger.bind(thread="InferenceWorker", device=str(self.device))
        log.info("Starting")

        while not self.stop_event.is_set():
            try:
                data = self.inference_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if data is None:  # Sentinel
                log.info("Sentinel received, shutting down")
                break

            t0 = time.perf_counter()
            alerts = self._infer_and_alert(data)
            latency_ms = (time.perf_counter() - t0) * 1000

            self._inference_count += 1
            self._latencies_ms.append(latency_ms)

            for alert in alerts:
                alert.inference_latency_ms = round(latency_ms, 2)
                self.alert_callback(alert)
                self._alert_count += 1

            if self._inference_count % 20 == 0:
                p50, p95 = self._percentile_latency()
                log.info(
                    "Stats",
                    inferences=self._inference_count,
                    alerts=self._alert_count,
                    latency_p50_ms=p50,
                    latency_p95_ms=p95,
                )

        log.info(
            "Done",
            total_inferences=self._inference_count,
            total_alerts=self._alert_count,
        )

    @torch.no_grad()
    def _infer_and_alert(self, data) -> list[Alert]:
        """Run forward pass, return list of Alert objects above threshold."""
        alerts = []

        try:
            data = data.to(self.device)

            # Model expects (x, edge_index, edge_attr). Training snapshots may
            # include the ground-truth botnet_fraction as the 8th edge feature;
            # realtime snapshots should already be stripped to 7 dimensions.
            if data.edge_attr is None:
                edge_attr = None
            elif data.edge_attr.shape[1] > 7:
                edge_attr = data.edge_attr[:, :7]
            else:
                edge_attr = data.edge_attr

            logits = self.model(data.x, data.edge_index, edge_attr)
            probs = torch.softmax(logits, dim=-1)  # [num_nodes, 2]

            node_ips: list[str] = getattr(data, "node_ips", [])
            graph_nodes = data.num_nodes
            graph_edges = data.num_edges
            timestamp = float(getattr(data, "timestamp", time.time()))

            for i, ip in enumerate(node_ips):
                botnet_prob = probs[i, 1].item()

                if botnet_prob < self.threshold:
                    continue

                # Deduplication check
                now = time.time()
                if ip in self._last_alert and now - self._last_alert[ip] < self.dedup_window_s:
                    continue

                self._last_alert[ip] = now

                reasons = self._explain(data, i, botnet_prob)
                dst_ip = self._top_neighbor_ip(data, i)

                alerts.append(
                    Alert(
                        timestamp=timestamp,
                        src_ip=ip,
                        dst_ip=dst_ip,
                        risk_score=round(botnet_prob, 4),
                        model=self.model.__class__.__name__,
                        reasons=reasons,
                        graph_version=int(getattr(data, "version", 0)),
                        inference_latency_ms=0.0,  # Set by caller
                        graph_nodes=graph_nodes,
                        graph_edges=graph_edges,
                    )
                )

        except Exception as exc:
            logger.error("Inference error", error=str(exc), exc_info=True)

        return alerts

    @staticmethod
    def _top_neighbor_ip(data, node_idx: int) -> str | None:
        """Best-effort peer IP for alert context."""
        node_ips: list[str] = getattr(data, "node_ips", [])
        if not node_ips or data.edge_index.numel() == 0:
            return None

        edge_index = data.edge_index.cpu()
        outgoing = edge_index[0] == node_idx
        incoming = edge_index[1] == node_idx

        if outgoing.any():
            peer_idx = int(edge_index[1][outgoing][0].item())
        elif incoming.any():
            peer_idx = int(edge_index[0][incoming][0].item())
        else:
            return None

        return node_ips[peer_idx] if 0 <= peer_idx < len(node_ips) else None

    def _explain(self, data, node_idx: int, score: float) -> list[str]:
        """
        Heuristic-based reason generation.
        TODO: Replace with proper GNN explainability (GNNExplainer or attention weights).
        """
        reasons = []
        x = data.x[node_idx]

        out_flows = x[1].item()
        unique_dsts = x[7].item()
        avg_dur = x[8].item()
        dst_entropy = x[11].item()
        suspicious_port_ratio = x[13].item()

        if unique_dsts > 5:
            reasons.append(f"high fan-out: {int(unique_dsts)} unique destinations")
        if avg_dur < 1.0:
            reasons.append("short-lived connections (possible beaconing)")
        if suspicious_port_ratio > 0.5:
            reasons.append(f"suspicious port ratio: {suspicious_port_ratio:.0%}")
        if out_flows > 50:
            reasons.append(f"high outbound flow count: {int(out_flows)}")
        if dst_entropy > 3.0:
            reasons.append("high destination entropy (scanning/C2 fan-out)")
        if score > 0.9:
            reasons.append("very high model confidence")

        return reasons or ["anomalous communication pattern detected by GNN"]

    def _percentile_latency(self) -> tuple[float, float]:
        if not self._latencies_ms:
            return 0.0, 0.0
        sorted_lat = sorted(self._latencies_ms[-100:])  # Last 100
        n = len(sorted_lat)
        p50 = sorted_lat[int(n * 0.5)]
        p95 = sorted_lat[int(n * 0.95)]
        return round(p50, 1), round(p95, 1)

    @staticmethod
    def _default_handler(alert: Alert) -> None:
        logger.warning("ALERT", alert=str(alert))
        Path("reports").mkdir(parents=True, exist_ok=True)
        with open("reports/alerts.jsonl", "a") as f:
            f.write(alert.to_json() + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — RealtimePipeline
# ─────────────────────────────────────────────────────────────────────────────


class RealtimePipeline:
    """
    Orchestrates the 3-thread realtime C2 detection pipeline.

    Usage:
        from c2gnn.models.graphsage import GraphSAGEC2Detector
        import torch

        model = GraphSAGEC2Detector()
        model.load_state_dict(torch.load("models/artifacts/graphsage_best.pt"))

        pipeline = RealtimePipeline(
            data_path=Path("data/processed/ctu13_scenario10_flows.parquet"),
            model=model,
            window_size=60.0,
            threshold=0.75,
            realtime_factor=20.0,
        )
        pipeline.run_until_complete()
    """

    def __init__(
        self,
        data_path: Path,
        model: nn.Module,
        window_size: float = 60.0,
        threshold: float = 0.7,
        realtime_factor: float = 10.0,
        snapshot_interval: float = 5.0,
        alert_callback: Callable[[Alert], None] | None = None,
        max_flows: int | None = None,
    ):
        self.stop_event = threading.Event()

        # Queues — sized to absorb burst without OOM
        self.flow_queue: queue.Queue = queue.Queue(maxsize=20_000)
        self.inference_queue: queue.Queue = queue.Queue(maxsize=200)

        self.flow_builder = FlowBuilderWorker(
            data_path=data_path,
            output_queue=self.flow_queue,
            realtime_factor=realtime_factor,
            max_flows=max_flows,
            stop_event=self.stop_event,
        )

        self.graph_updater = GraphUpdateWorker(
            flow_queue=self.flow_queue,
            inference_queue=self.inference_queue,
            window_size=window_size,
            edge_ttl=window_size * 2,
            snapshot_interval=snapshot_interval,
            stop_event=self.stop_event,
        )

        self.inference_worker = InferenceWorker(
            inference_queue=self.inference_queue,
            model=model,
            threshold=threshold,
            alert_callback=alert_callback,
            stop_event=self.stop_event,
        )

    def start(self) -> None:
        """Start all 3 threads."""
        logger.info("Pipeline starting", threads=3)
        self.flow_builder.start()
        self.graph_updater.start()
        self.inference_worker.start()

    def stop(self) -> None:
        """Signal all threads to stop gracefully."""
        logger.info("Pipeline stopping")
        self.stop_event.set()

    def join(self, timeout: float = 60.0) -> None:
        for worker in [self.flow_builder, self.graph_updater, self.inference_worker]:
            worker.join(timeout=timeout)

    def run_until_complete(self) -> None:
        """Start pipeline and block until FlowBuilder finishes."""
        self.start()
        try:
            # Block on FlowBuilder completion (it sends sentinel when done)
            self.flow_builder.join()
            self.graph_updater.join()
            self.inference_worker.join()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self.stop()
            self.join(timeout=5.0)

    @property
    def stats(self) -> dict:
        return {
            "flow_queue_size": self.flow_queue.qsize(),
            "inference_queue_size": self.inference_queue.qsize(),
            "flow_builder_stats": self.flow_builder.stats,
            "graph_stats": self.graph_updater.graph.stats,
            "inference_count": self.inference_worker._inference_count,
            "alert_count": self.inference_worker._alert_count,
        }


def _load_model(model_path: Path, model_type: str | None = None) -> nn.Module:
    from c2gnn.models.graphsage import GATv2C2Detector, GraphSAGEC2Detector

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    inferred_type = model_type or (
        checkpoint.get("model_type") if isinstance(checkpoint, dict) else None
    )
    inferred_type = inferred_type or ("gatv2" if "gat" in model_path.name.lower() else "graphsage")

    if inferred_type.lower() in {"gat", "gatv2"}:
        model = GATv2C2Detector()
    elif inferred_type.lower() in {"sage", "graphsage"}:
        model = GraphSAGEC2Detector(hidden_channels=128)
    else:
        raise ValueError(f"Unsupported model type: {inferred_type}")

    model.load_state_dict(state_dict)
    model.eval()
    return model


def _api_alert_callback(api_url: str) -> Callable[[Alert], None]:
    import requests

    endpoint = api_url.rstrip("/") + "/api/v1/alerts"

    def post_alert(alert: Alert) -> None:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(alert.timestamp)),
            "src_ip": alert.src_ip,
            "dst_ip": alert.dst_ip or "unknown",
            "risk_score": alert.risk_score,
            "model": alert.model,
            "reason": alert.reasons,
            "graph_stats": {
                "version": alert.graph_version,
                "nodes": alert.graph_nodes,
                "edges": alert.graph_edges,
                "latency_ms": alert.inference_latency_ms,
            },
        }
        try:
            requests.post(endpoint, json=payload, timeout=2).raise_for_status()
        except Exception as exc:
            logger.warning("alert_api_post_failed", error=str(exc), endpoint=endpoint)
            InferenceWorker._default_handler(alert)

    return post_alert


def main() -> None:
    parser = argparse.ArgumentParser(description="Run realtime C2 detection replay")
    parser.add_argument("--data", required=True, type=Path, help="Input .parquet/.binetflow file")
    parser.add_argument("--model", required=True, type=Path, help="Trained .pt model checkpoint")
    parser.add_argument("--model-type", choices=["graphsage", "gatv2"], default=None)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--window-size", type=float, default=60.0)
    parser.add_argument("--realtime-factor", type=float, default=50.0)
    parser.add_argument("--snapshot-interval", type=float, default=5.0)
    parser.add_argument("--max-flows", type=int, default=None)
    parser.add_argument("--api-url", default=None, help="Optional Alert API base URL")
    args = parser.parse_args()

    model = _load_model(args.model, args.model_type)
    callback = _api_alert_callback(args.api_url) if args.api_url else None
    pipeline = RealtimePipeline(
        data_path=args.data,
        model=model,
        window_size=args.window_size,
        threshold=args.threshold,
        realtime_factor=args.realtime_factor,
        snapshot_interval=args.snapshot_interval,
        alert_callback=callback,
        max_flows=args.max_flows,
    )
    pipeline.run_until_complete()
    logger.info("pipeline_complete", **pipeline.stats)


if __name__ == "__main__":
    main()
