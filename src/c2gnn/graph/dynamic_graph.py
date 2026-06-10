"""
dynamic_graph.py — Sliding Window Dynamic Graph

Xây dựng graph động từ network flows với:
  - Sliding window theo thời gian thực
  - TTL-based edge expiry (không rebuild toàn bộ)
  - Incremental node feature update
  - Export sang PyTorch Geometric Data

Author: Member 1 (Data/Network/Security Engineer) + Member 2 (AI Engineer)
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import structlog
import torch
from torch_geometric.data import Data

from c2gnn.data.flow_builder import FlowRecord

logger = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Edge and Node data containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EdgeData:
    """
    Aggregated statistics for a directed edge (src_ip → dst_ip).
    Updated incrementally on each new flow.
    """

    src_ip: str
    dst_ip: str
    flow_count: int = 0
    total_bytes: int = 0
    total_packets: int = 0
    total_duration: float = 0.0
    first_seen: float = float("inf")
    last_seen: float = 0.0
    labels: list = field(default_factory=list)
    src_ports: list = field(default_factory=list)
    dst_ports: list = field(default_factory=list)
    protocols: list = field(default_factory=list)

    def update(self, flow: FlowRecord) -> None:
        """Incremental update — O(1)."""
        self.flow_count += 1
        self.total_bytes += flow.total_bytes
        self.total_packets += flow.total_fwd_packets + flow.total_bwd_packets
        self.total_duration += flow.duration
        self.first_seen = min(self.first_seen, flow.timestamp)
        self.last_seen = max(self.last_seen, flow.timestamp)
        self.labels.append(flow.label)
        if flow.src_port:
            self.src_ports.append(flow.src_port)
        if flow.dst_port:
            self.dst_ports.append(flow.dst_port)
        self.protocols.append(flow.protocol)

    # ── Derived properties ──────────────────────────────────────────────────

    @property
    def avg_duration(self) -> float:
        return self.total_duration / max(self.flow_count, 1)

    @property
    def avg_bytes_per_flow(self) -> float:
        return self.total_bytes / max(self.flow_count, 1)

    @property
    def avg_packets_per_flow(self) -> float:
        return self.total_packets / max(self.flow_count, 1)

    @property
    def existence_duration(self) -> float:
        """How long this edge has been present in the window."""
        return self.last_seen - self.first_seen

    @property
    def label(self) -> str:
        """Majority vote label."""
        if not self.labels:
            return "background"
        return collections.Counter(self.labels).most_common(1)[0][0]

    @property
    def is_botnet(self) -> bool:
        return self.label == "botnet"

    @property
    def botnet_fraction(self) -> float:
        if not self.labels:
            return 0.0
        return self.labels.count("botnet") / len(self.labels)

    @staticmethod
    def _port_entropy(ports: list[int]) -> float:
        if not ports:
            return 0.0
        counts = collections.Counter(ports)
        total = len(ports)
        return -sum((c / total) * math.log2(c / total + 1e-12) for c in counts.values())

    @property
    def dst_port_entropy(self) -> float:
        return self._port_entropy(self.dst_ports)

    def to_feature_vector(self) -> np.ndarray:
        """Edge feature vector for PyG edge_attr."""
        return np.array(
            [
                float(self.flow_count),
                float(self.total_bytes),
                float(self.total_packets),
                float(self.avg_duration),
                float(self.avg_bytes_per_flow),
                float(self.dst_port_entropy),
                float(self.existence_duration),
                float(self.botnet_fraction),  # ground truth only — zero out for inference
            ],
            dtype=np.float32,
        )

    @classmethod
    def feature_names(cls) -> list[str]:
        return [
            "flow_count",
            "total_bytes",
            "total_packets",
            "avg_duration",
            "avg_bytes_per_flow",
            "dst_port_entropy",
            "existence_duration",
            "botnet_fraction",
        ]


@dataclass
class NodeData:
    """
    Aggregated per-node (IP) statistics within the current window.
    Updated incrementally as flows arrive.
    """

    ip: str
    in_bytes: int = 0
    out_bytes: int = 0
    in_flows: int = 0
    out_flows: int = 0
    in_packets: int = 0
    out_packets: int = 0
    unique_srcs: set = field(default_factory=set)
    unique_dsts: set = field(default_factory=set)
    durations: list = field(default_factory=list)
    dst_ports_seen: list = field(default_factory=list)
    src_ports_seen: list = field(default_factory=list)
    protocols_seen: list = field(default_factory=list)
    last_seen: float = 0.0

    def update_as_src(self, flow: FlowRecord) -> None:
        self.out_bytes += flow.total_bytes
        self.out_flows += 1
        self.out_packets += flow.total_fwd_packets
        self.unique_dsts.add(flow.dst_ip)
        self.durations.append(flow.duration)
        if flow.dst_port:
            self.dst_ports_seen.append(flow.dst_port)
        self.protocols_seen.append(flow.protocol)
        self.last_seen = max(self.last_seen, flow.timestamp)

    def update_as_dst(self, flow: FlowRecord) -> None:
        self.in_bytes += flow.total_bytes
        self.in_flows += 1
        self.in_packets += flow.total_bwd_packets
        self.unique_srcs.add(flow.src_ip)
        if flow.src_port:
            self.src_ports_seen.append(flow.src_port)
        self.last_seen = max(self.last_seen, flow.timestamp)

    # ── Derived ──────────────────────────────────────────────────────────────

    @property
    def total_flows(self) -> int:
        return self.in_flows + self.out_flows

    @property
    def total_bytes(self) -> int:
        return self.in_bytes + self.out_bytes

    @property
    def avg_duration(self) -> float:
        return sum(self.durations) / len(self.durations) if self.durations else 0.0

    @property
    def std_duration(self) -> float:
        if len(self.durations) < 2:
            return 0.0
        mean = self.avg_duration
        return math.sqrt(sum((d - mean) ** 2 for d in self.durations) / len(self.durations))

    @property
    def fan_out_ratio(self) -> float:
        """High fan-out = talking to many unique destinations = suspicious."""
        return self.out_flows / max(self.in_flows, 1)

    @property
    def dst_ip_entropy(self) -> float:
        """Entropy over destination IPs — high = scanning/C2 fan-out."""
        n = len(self.unique_dsts)
        return math.log2(n + 1)

    @property
    def dst_port_entropy(self) -> float:
        if not self.dst_ports_seen:
            return 0.0
        counts = collections.Counter(self.dst_ports_seen)
        total = len(self.dst_ports_seen)
        return -sum((c / total) * math.log2(c / total + 1e-12) for c in counts.values())

    @property
    def suspicious_port_ratio(self) -> float:
        """Ratio of connections to unusual/high ports."""
        if not self.dst_ports_seen:
            return 0.0
        unusual = sum(1 for p in self.dst_ports_seen if p >= 49152 or 1024 <= p <= 1099)
        return unusual / len(self.dst_ports_seen)

    def to_feature_vector(self) -> np.ndarray:
        """
        14-dimensional node feature vector for GNN input.
        IMPORTANT: Keep this in sync with NODE_FEATURE_DIM in graphsage.py
        """
        return np.array(
            [
                float(self.in_flows),  # 0
                float(self.out_flows),  # 1
                float(self.total_flows),  # 2
                float(self.in_bytes),  # 3
                float(self.out_bytes),  # 4
                float(self.total_bytes),  # 5
                float(len(self.unique_srcs)),  # 6
                float(len(self.unique_dsts)),  # 7
                float(self.avg_duration),  # 8
                float(self.std_duration),  # 9
                float(self.fan_out_ratio),  # 10
                float(self.dst_ip_entropy),  # 11
                float(self.dst_port_entropy),  # 12
                float(self.suspicious_port_ratio),  # 13
            ],
            dtype=np.float32,
        )

    @classmethod
    def feature_names(cls) -> list[str]:
        return [
            "in_flows",
            "out_flows",
            "total_flows",
            "in_bytes",
            "out_bytes",
            "total_bytes",
            "unique_src_count",
            "unique_dst_count",
            "avg_duration",
            "std_duration",
            "fan_out_ratio",
            "dst_ip_entropy",
            "dst_port_entropy",
            "suspicious_port_ratio",
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Sliding Window Dynamic Graph
# ─────────────────────────────────────────────────────────────────────────────


class SlidingWindowGraph:
    """
    Dynamic graph with sliding time window and TTL-based edge expiry.

    Design decisions:
    1. Incremental updates: O(1) per flow vs O(n) full rebuild
    2. TTL queue: min-heap style deque for efficient expiry check
    3. No deep copies: NodeData/EdgeData mutated in-place
    4. PyG conversion: only on demand (snapshot_interval controlled by caller)

    Args:
        window_size: How far back to keep data (seconds). Flows older than
                     current_time - window_size are no longer considered fresh.
        edge_ttl: Seconds after last_seen before an edge is evicted.
                  Should be >= window_size for consistent windows.
        min_flows_per_edge: Edges with fewer flows are excluded from PyG export.
    """

    def __init__(
        self,
        window_size: float = 60.0,
        edge_ttl: float = 120.0,
        min_flows_per_edge: int = 1,
    ):
        self.window_size = window_size
        self.edge_ttl = edge_ttl
        self.min_flows_per_edge = min_flows_per_edge

        # Core data stores
        self._node_data: dict[str, NodeData] = {}
        self._edge_data: dict[tuple[str, str], EdgeData] = {}

        # NetworkX graph for topology operations (fast neighbor queries)
        self._graph: nx.DiGraph = nx.DiGraph()

        # TTL queue: (expiry_timestamp, (src_ip, dst_ip))
        # Not a true priority queue — we just scan front (flows are time-ordered)
        self._expiry_queue: collections.deque = collections.deque()

        self._current_time: float = 0.0
        self._version: int = 0  # Incremented on every update
        self._flows_processed: int = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def update(self, flow: FlowRecord) -> list[str]:
        """
        Incremental graph update with a new flow.

        Returns: List of node IPs that were modified (for targeted re-inference).
        Complexity: O(expired_edges) amortized, O(1) for the common case.
        """
        self._current_time = max(self._current_time, flow.timestamp)
        self._flows_processed += 1

        # 1. Expire stale edges first
        self._expire_old_edges()

        # 2. Update nodes
        for ip in (flow.src_ip, flow.dst_ip):
            if ip not in self._node_data:
                self._node_data[ip] = NodeData(ip=ip)
                self._graph.add_node(ip)

        self._node_data[flow.src_ip].update_as_src(flow)
        self._node_data[flow.dst_ip].update_as_dst(flow)

        # 3. Update or create edge
        edge_key = (flow.src_ip, flow.dst_ip)
        if edge_key not in self._edge_data:
            self._edge_data[edge_key] = EdgeData(
                src_ip=flow.src_ip,
                dst_ip=flow.dst_ip,
            )

        self._edge_data[edge_key].update(flow)

        if not self._graph.has_edge(flow.src_ip, flow.dst_ip):
            self._graph.add_edge(flow.src_ip, flow.dst_ip)

        # 4. Register expiry (TTL from last_seen)
        expiry_ts = flow.timestamp + self.edge_ttl
        self._expiry_queue.append((expiry_ts, edge_key))

        self._version += 1
        return [flow.src_ip, flow.dst_ip]

    def to_pyg_data(self, include_ground_truth: bool = True) -> Data | None:
        """
        Snapshot the current graph as a PyTorch Geometric Data object.

        Args:
            include_ground_truth: If False, strips botnet_fraction from edge_attr
                                  (for true inference mode, not training).

        Returns: PyG Data or None if graph is empty.
        """
        nodes = list(self._graph.nodes())
        if not nodes:
            return None

        # Filter edges by min_flows
        edges = [
            (s, d)
            for s, d in self._graph.edges()
            if self._edge_data.get((s, d), EdgeData(s, d)).flow_count >= self.min_flows_per_edge
        ]

        if not edges:
            return None

        node_to_idx = {ip: i for i, ip in enumerate(nodes)}

        # ── Node features ───────────────────────────────────────────────────
        x_list = []
        y_list = []

        for ip in nodes:
            nd = self._node_data.get(ip, NodeData(ip=ip))
            x_list.append(nd.to_feature_vector())

            # Node label: 1 if this IP has any botnet edge in the current window
            if include_ground_truth:
                is_bot = any(
                    self._edge_data.get((ip, nbr), EdgeData(ip, nbr)).is_botnet
                    for nbr in self._graph.successors(ip)
                ) or any(
                    self._edge_data.get((nbr, ip), EdgeData(nbr, ip)).is_botnet
                    for nbr in self._graph.predecessors(ip)
                )
                y_list.append(1 if is_bot else 0)

        x = torch.tensor(np.stack(x_list), dtype=torch.float32)

        # ── Edge index & attributes ─────────────────────────────────────────
        src_idx = [node_to_idx[s] for s, _ in edges]
        dst_idx = [node_to_idx[d] for _, d in edges]
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

        edge_attr_list = []
        for s, d in edges:
            ed = self._edge_data.get((s, d), EdgeData(s, d))
            feat = ed.to_feature_vector()
            if not include_ground_truth:
                feat = feat[:-1]  # Strip botnet_fraction
            edge_attr_list.append(feat)

        edge_attr = torch.tensor(np.stack(edge_attr_list), dtype=torch.float32)

        # ── Assemble Data ───────────────────────────────────────────────────
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if include_ground_truth and y_list:
            data.y = torch.tensor(y_list, dtype=torch.long)
        elif not include_ground_truth:
            data.y = torch.zeros(len(nodes), dtype=torch.long)

        # Metadata (not used by GNN, used for alerts)
        data.node_ips = nodes
        data.edge_pairs = edges
        data.version = self._version
        data.timestamp = self._current_time
        data.num_botnet_nodes = sum(y_list) if y_list else 0

        return data

    def get_suspicious_subgraph(self, ip: str, hops: int = 2) -> Data | None:
        """
        Extract k-hop subgraph around a suspicious IP for targeted inference.
        More efficient than full graph inference on large graphs.
        """
        if ip not in self._graph:
            return None

        neighbors = nx.ego_graph(self._graph, ip, radius=hops)
        subgraph = SlidingWindowGraph(window_size=self.window_size, edge_ttl=self.edge_ttl)

        for node in neighbors.nodes():
            subgraph._graph.add_node(node)
            if node in self._node_data:
                subgraph._node_data[node] = self._node_data[node]

        for s, d in neighbors.edges():
            subgraph._graph.add_edge(s, d)
            if (s, d) in self._edge_data:
                subgraph._edge_data[(s, d)] = self._edge_data[(s, d)]

        subgraph._current_time = self._current_time
        return subgraph.to_pyg_data()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _expire_old_edges(self) -> int:
        """
        Evict edges where last_seen < current_time - edge_ttl.
        Returns number of edges removed.
        """
        cutoff = self._current_time - self.edge_ttl
        removed = 0

        while self._expiry_queue:
            expiry_ts, edge_key = self._expiry_queue[0]

            if expiry_ts > self._current_time:
                break  # Queue is time-ordered, nothing else to expire

            self._expiry_queue.popleft()
            src, dst = edge_key

            if edge_key in self._edge_data and self._edge_data[edge_key].last_seen < cutoff:
                self._remove_edge(edge_key)
                removed += 1

        return removed

    def _remove_edge(self, edge_key: tuple[str, str]) -> None:
        src, dst = edge_key

        if self._graph.has_edge(src, dst):
            self._graph.remove_edge(src, dst)

        self._edge_data.pop(edge_key, None)

        # Remove isolated nodes to keep graph clean
        for ip in (src, dst):
            if ip in self._graph and self._graph.degree(ip) == 0:
                self._graph.remove_node(ip)
                self._node_data.pop(ip, None)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    @property
    def stats(self) -> dict:
        bot_nodes = sum(
            1
            for ip in self._node_data
            if any(
                self._edge_data.get((ip, nbr), EdgeData(ip, nbr)).is_botnet
                for nbr in self._graph.successors(ip)
            )
        )
        return {
            "version": self._version,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "flows_processed": self._flows_processed,
            "current_time": self._current_time,
            "botnet_nodes_in_window": bot_nodes,
        }
