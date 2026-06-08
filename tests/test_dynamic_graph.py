"""
Tests for c2gnn.graph.dynamic_graph

Covers: EdgeData stats, NodeData stats, SlidingWindowGraph
        (incremental update, TTL expiry, PyG export).
"""
import math
import time

import numpy as np
import pytest

from c2gnn.graph.dynamic_graph import EdgeData, NodeData, SlidingWindowGraph
from tests.conftest import make_flow


# ─────────────────────────────────────────────────────────────────────────────
# EdgeData
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeData:
    def test_update_increments_flow_count(self):
        ed = EdgeData(src_ip="A", dst_ip="B")
        flow = make_flow("A", "B", total_bytes=100)
        ed.update(flow)
        assert ed.flow_count == 1

    def test_update_accumulates_bytes(self):
        ed = EdgeData(src_ip="A", dst_ip="B")
        ed.update(make_flow("A", "B", total_bytes=100))
        ed.update(make_flow("A", "B", total_bytes=200))
        assert ed.total_bytes == 300

    def test_avg_bytes_per_flow(self):
        ed = EdgeData(src_ip="A", dst_ip="B")
        ed.update(make_flow("A", "B", total_bytes=100))
        ed.update(make_flow("A", "B", total_bytes=300))
        assert ed.avg_bytes_per_flow == 200.0

    def test_avg_duration(self):
        ed = EdgeData(src_ip="A", dst_ip="B")
        ed.update(make_flow("A", "B", duration=2.0))
        ed.update(make_flow("A", "B", duration=4.0))
        assert ed.avg_duration == 3.0

    def test_first_last_seen(self):
        ed = EdgeData(src_ip="A", dst_ip="B")
        ed.update(make_flow("A", "B", timestamp=1000.0))
        ed.update(make_flow("A", "B", timestamp=1010.0))
        assert ed.first_seen == 1000.0
        assert ed.last_seen == 1010.0

    def test_existence_duration(self):
        ed = EdgeData(src_ip="A", dst_ip="B")
        ed.update(make_flow("A", "B", timestamp=1000.0))
        ed.update(make_flow("A", "B", timestamp=1020.0))
        assert ed.existence_duration == 20.0

    def test_label_majority_vote_botnet(self):
        ed = EdgeData(src_ip="A", dst_ip="B")
        ed.update(make_flow("A", "B", label="botnet"))
        ed.update(make_flow("A", "B", label="botnet"))
        ed.update(make_flow("A", "B", label="normal"))
        assert ed.label == "botnet"
        assert ed.is_botnet is True

    def test_label_majority_vote_normal(self):
        ed = EdgeData(src_ip="A", dst_ip="B")
        ed.update(make_flow("A", "B", label="normal"))
        ed.update(make_flow("A", "B", label="normal"))
        ed.update(make_flow("A", "B", label="botnet"))
        assert ed.label == "normal"
        assert ed.is_botnet is False

    def test_botnet_fraction(self):
        ed = EdgeData(src_ip="A", dst_ip="B")
        for _ in range(3):
            ed.update(make_flow("A", "B", label="botnet"))
        for _ in range(7):
            ed.update(make_flow("A", "B", label="normal"))
        assert abs(ed.botnet_fraction - 0.3) < 1e-9

    def test_dst_port_entropy_single_port_is_zero(self):
        ed = EdgeData(src_ip="A", dst_ip="B")
        for _ in range(5):
            ed.update(make_flow("A", "B", dst_port=80))
        # Single unique port → entropy ≈ 0
        assert ed.dst_port_entropy < 0.1

    def test_dst_port_entropy_many_ports_is_high(self):
        ed = EdgeData(src_ip="A", dst_ip="B")
        for port in range(1, 20):
            ed.update(make_flow("A", "B", dst_port=port))
        assert ed.dst_port_entropy > 3.0

    def test_to_feature_vector_shape(self):
        ed = EdgeData(src_ip="A", dst_ip="B")
        ed.update(make_flow("A", "B"))
        vec = ed.to_feature_vector()
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (8,)
        assert vec.dtype == np.float32

    def test_feature_names_length_matches_vector(self):
        vec = EdgeData(src_ip="A", dst_ip="B").to_feature_vector()
        assert len(EdgeData.feature_names()) == len(vec)


# ─────────────────────────────────────────────────────────────────────────────
# NodeData
# ─────────────────────────────────────────────────────────────────────────────


class TestNodeData:
    def test_update_as_src_increments_out_flows(self):
        nd = NodeData(ip="10.0.0.1")
        nd.update_as_src(make_flow("10.0.0.1", "10.0.0.2"))
        assert nd.out_flows == 1
        assert nd.in_flows == 0

    def test_update_as_dst_increments_in_flows(self):
        nd = NodeData(ip="10.0.0.2")
        nd.update_as_dst(make_flow("10.0.0.1", "10.0.0.2"))
        assert nd.in_flows == 1
        assert nd.out_flows == 0

    def test_total_flows(self):
        nd = NodeData(ip="10.0.0.1")
        nd.update_as_src(make_flow("10.0.0.1", "10.0.0.2"))
        nd.update_as_dst(make_flow("10.0.0.3", "10.0.0.1"))
        assert nd.total_flows == 2

    def test_unique_dsts_tracked(self):
        nd = NodeData(ip="10.0.0.1")
        nd.update_as_src(make_flow("10.0.0.1", "10.0.0.2"))
        nd.update_as_src(make_flow("10.0.0.1", "10.0.0.3"))
        nd.update_as_src(make_flow("10.0.0.1", "10.0.0.2"))  # duplicate
        assert len(nd.unique_dsts) == 2

    def test_fan_out_ratio_high(self):
        nd = NodeData(ip="10.0.0.1")
        for i in range(10):
            nd.update_as_src(make_flow("10.0.0.1", f"10.0.0.{i}"))
        # in_flows = 0 → ratio = out_flows / 1 = 10
        assert nd.fan_out_ratio == 10.0

    def test_avg_and_std_duration(self):
        nd = NodeData(ip="A")
        nd.update_as_src(make_flow("A", "B", duration=2.0))
        nd.update_as_src(make_flow("A", "C", duration=4.0))
        assert nd.avg_duration == 3.0
        # std = sqrt(((2-3)^2 + (4-3)^2) / 2) = 1.0
        assert abs(nd.std_duration - 1.0) < 1e-9

    def test_std_duration_single_flow_is_zero(self):
        nd = NodeData(ip="A")
        nd.update_as_src(make_flow("A", "B", duration=5.0))
        assert nd.std_duration == 0.0

    def test_dst_ip_entropy_increases_with_unique_dsts(self):
        nd1 = NodeData(ip="A")
        nd1.update_as_src(make_flow("A", "B"))

        nd2 = NodeData(ip="A")
        for i in range(10):
            nd2.update_as_src(make_flow("A", f"dst{i}"))

        assert nd2.dst_ip_entropy > nd1.dst_ip_entropy

    def test_suspicious_port_ratio_high_ports(self):
        nd = NodeData(ip="A")
        for port in range(49152, 49162):  # all high ephemeral ports
            nd.update_as_src(make_flow("A", "B", dst_port=port))
        assert nd.suspicious_port_ratio == 1.0

    def test_suspicious_port_ratio_well_known_ports(self):
        nd = NodeData(ip="A")
        for port in [80, 443, 22, 25, 53]:
            nd.update_as_src(make_flow("A", "B", dst_port=port))
        assert nd.suspicious_port_ratio == 0.0

    def test_to_feature_vector_shape(self):
        nd = NodeData(ip="A")
        nd.update_as_src(make_flow("A", "B"))
        vec = nd.to_feature_vector()
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (14,)
        assert vec.dtype == np.float32

    def test_feature_names_length_matches_vector(self):
        assert len(NodeData.feature_names()) == 14


# ─────────────────────────────────────────────────────────────────────────────
# SlidingWindowGraph
# ─────────────────────────────────────────────────────────────────────────────


class TestSlidingWindowGraph:
    def test_update_creates_nodes(self):
        g = SlidingWindowGraph()
        g.update(make_flow("A", "B", timestamp=1000.0))
        assert "A" in g._node_data
        assert "B" in g._node_data

    def test_update_creates_edge(self):
        g = SlidingWindowGraph()
        g.update(make_flow("A", "B", timestamp=1000.0))
        assert ("A", "B") in g._edge_data

    def test_update_returns_modified_ips(self):
        g = SlidingWindowGraph()
        modified = g.update(make_flow("A", "B", timestamp=1000.0))
        assert "A" in modified
        assert "B" in modified

    def test_multiple_flows_same_edge_increments_count(self):
        g = SlidingWindowGraph()
        g.update(make_flow("A", "B", timestamp=1000.0))
        g.update(make_flow("A", "B", timestamp=1001.0))
        g.update(make_flow("A", "B", timestamp=1002.0))
        assert g._edge_data[("A", "B")].flow_count == 3

    def test_version_increments_on_update(self):
        g = SlidingWindowGraph()
        v0 = g._version
        g.update(make_flow("A", "B", timestamp=1000.0))
        assert g._version == v0 + 1

    def test_flows_processed_count(self):
        g = SlidingWindowGraph()
        for i in range(5):
            g.update(make_flow("A", "B", timestamp=float(1000 + i)))
        assert g._flows_processed == 5

    def test_edge_expiry_removes_stale_edges(self):
        g = SlidingWindowGraph(window_size=60.0, edge_ttl=10.0)
        g.update(make_flow("A", "B", timestamp=1000.0))
        assert ("A", "B") in g._edge_data

        # Advance time past TTL — add a flow far in the future
        g.update(make_flow("C", "D", timestamp=1020.0))  # 20s later > ttl=10

        # Edge A→B should have expired
        assert ("A", "B") not in g._edge_data

    def test_edge_expiry_keeps_fresh_edges(self):
        g = SlidingWindowGraph(window_size=60.0, edge_ttl=30.0)
        g.update(make_flow("A", "B", timestamp=1000.0))
        g.update(make_flow("A", "B", timestamp=1010.0))  # refresh last_seen

        # Only 15s later — within TTL of last refresh
        g.update(make_flow("C", "D", timestamp=1015.0))

        assert ("A", "B") in g._edge_data

    def test_to_pyg_data_returns_none_on_empty_graph(self):
        g = SlidingWindowGraph()
        assert g.to_pyg_data() is None

    def test_to_pyg_data_node_features_shape(self, small_graph):
        data = small_graph.to_pyg_data(include_ground_truth=False)
        assert data is not None
        # 3 unique IPs: 10.0.0.1, 10.0.0.2, 10.0.0.3
        assert data.x.shape == (3, 14)

    def test_to_pyg_data_edge_index_shape(self, small_graph):
        data = small_graph.to_pyg_data(include_ground_truth=False)
        assert data is not None
        # 3 flows → 3 directed edges
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == 3

    def test_to_pyg_data_with_ground_truth_has_y(self, small_graph):
        data = small_graph.to_pyg_data(include_ground_truth=True)
        assert data is not None
        assert data.y is not None
        assert data.y.shape[0] == data.x.shape[0]

    def test_to_pyg_data_without_ground_truth_still_has_y(self, small_graph):
        # Even without GT, y is set to zeros
        data = small_graph.to_pyg_data(include_ground_truth=False)
        assert data is not None
        assert data.y is not None

    def test_botnet_node_labeled_correctly(self):
        g = SlidingWindowGraph()
        g.update(make_flow("bot", "c2server", timestamp=1000.0, label="botnet"))
        g.update(make_flow("normal", "server", timestamp=1000.0, label="normal"))
        data = g.to_pyg_data(include_ground_truth=True)
        assert data is not None

        nodes = list(g._graph.nodes())
        node_idx = {ip: i for i, ip in enumerate(nodes)}

        bot_idx = node_idx.get("bot")
        if bot_idx is not None:
            assert data.y[bot_idx].item() == 1

    def test_min_flows_filter_excludes_edges(self):
        g = SlidingWindowGraph(min_flows_per_edge=3)
        # Only 1 flow on this edge → should be excluded from PyG export
        g.update(make_flow("A", "B", timestamp=1000.0))
        # But 3 flows on this edge → included
        for i in range(3):
            g.update(make_flow("C", "D", timestamp=float(1000 + i)))

        data = g.to_pyg_data(include_ground_truth=False)
        assert data is not None
        # Only C→D edge should appear
        assert data.edge_index.shape[1] == 1

    def test_current_time_advances_monotonically(self):
        g = SlidingWindowGraph()
        g.update(make_flow("A", "B", timestamp=1000.0))
        g.update(make_flow("A", "B", timestamp=999.0))  # out-of-order
        # Should not regress
        assert g._current_time == 1000.0
