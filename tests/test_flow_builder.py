"""
Tests for c2gnn.data.flow_builder

Covers: FlowRecord properties, label normalization, timestamp parsing,
        port parsing, BeaconingDetector scoring.
"""

import time

import pytest

from c2gnn.data.flow_builder import (
    BeaconingDetector,
    _normalize_label,
    _parse_port,
    _parse_timestamp,
)
from tests.conftest import make_flow

# ─────────────────────────────────────────────────────────────────────────────
# FlowRecord — derived properties
# ─────────────────────────────────────────────────────────────────────────────


class TestFlowRecord:
    def test_is_malicious_true_for_botnet(self):
        flow = make_flow(label="botnet")
        assert flow.is_malicious is True

    def test_is_malicious_false_for_normal(self):
        flow = make_flow(label="normal")
        assert flow.is_malicious is False

    def test_is_background(self):
        flow = make_flow(label="background")
        assert flow.is_background is True
        assert flow.is_malicious is False

    def test_total_packets_sums_fwd_bwd(self):
        flow = make_flow(total_fwd_packets=7, total_bwd_packets=3)
        assert flow.total_packets == 10

    def test_bytes_per_packet_no_zero_division(self):
        # total_packets = 0 should not raise
        flow = make_flow(total_fwd_packets=0, total_bwd_packets=0, total_bytes=0)
        assert flow.bytes_per_packet == 0.0

    def test_bytes_per_packet_correct(self):
        flow = make_flow(total_fwd_packets=4, total_bwd_packets=4, total_bytes=800)
        assert flow.bytes_per_packet == 100.0

    def test_dst_port_well_known_http(self):
        flow = make_flow(dst_port=80)
        assert flow.dst_port_is_well_known is True

    def test_dst_port_not_well_known_high(self):
        flow = make_flow(dst_port=6667)
        assert flow.dst_port_is_well_known is False

    def test_dst_port_not_well_known_zero(self):
        flow = make_flow(dst_port=0)
        assert flow.dst_port_is_well_known is False

    def test_to_feature_dict_has_required_keys(self):
        flow = make_flow()
        feat = flow.to_feature_dict()
        required = {
            "duration",
            "total_fwd_packets",
            "total_bwd_packets",
            "total_packets",
            "total_bytes",
            "packet_rate",
            "byte_rate",
            "is_tcp",
            "is_udp",
            "is_icmp",
            "label_binary",
        }
        assert required.issubset(feat.keys())

    def test_to_feature_dict_label_binary_botnet(self):
        flow = make_flow(label="botnet")
        assert flow.to_feature_dict()["label_binary"] == 1.0

    def test_to_feature_dict_label_binary_normal(self):
        flow = make_flow(label="normal")
        assert flow.to_feature_dict()["label_binary"] == 0.0

    def test_to_feature_dict_protocol_flags(self):
        tcp_flow = make_flow(protocol="TCP")
        udp_flow = make_flow(protocol="UDP")
        icmp_flow = make_flow(protocol="ICMP")

        assert tcp_flow.to_feature_dict()["is_tcp"] == 1.0
        assert tcp_flow.to_feature_dict()["is_udp"] == 0.0

        assert udp_flow.to_feature_dict()["is_udp"] == 1.0
        assert icmp_flow.to_feature_dict()["is_icmp"] == 1.0

    def test_frozen_immutability(self):
        flow = make_flow()
        with pytest.raises((AttributeError, TypeError)):
            flow.label = "botnet"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Label normalization
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalizeLabel:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("flow=Background", "background"),
            ("flow=From-Background", "background"),
            ("flow=Normal", "normal"),
            ("flow=From-Botnet-V42-UDP", "botnet"),
            ("flow=To-Botnet-V42-TCP-CC", "botnet"),
            ("Botnet", "botnet"),
            ("BOTNET", "botnet"),
            ("", "background"),
            ("flow=Unknown-xyz", "background"),  # fallback
        ],
    )
    def test_normalize(self, raw, expected):
        assert _normalize_label(raw) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp parsing
# ─────────────────────────────────────────────────────────────────────────────


class TestParseTimestamp:
    def test_ctu13_format_with_microseconds(self):
        ts = _parse_timestamp("2011/08/10 09:46:53.047925")
        # Should be a float around 2011 epoch time
        assert ts > 1_300_000_000  # after ~2011

    def test_no_fractional(self):
        ts = _parse_timestamp("2011/08/10 09:46:53")
        assert ts > 1_300_000_000

    def test_empty_string_returns_current_time(self):
        before = time.time()
        ts = _parse_timestamp("")
        after = time.time()
        assert before <= ts <= after + 1

    def test_plain_float_string(self):
        ts = _parse_timestamp("1312970813.047")
        assert abs(ts - 1312970813.047) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# Port parsing
# ─────────────────────────────────────────────────────────────────────────────


class TestParsePort:
    @pytest.mark.parametrize(
        "val,expected",
        [
            ("80", 80),
            ("0x50", 80),  # hex
            ("0X1F90", 8080),  # hex uppercase
            ("443.0", 443),  # float string
            ("", 0),  # empty
            ("abc", 0),  # unparseable
            (0, 0),  # integer zero
            (None, 0),  # None
            ("65535", 65535),
        ],
    )
    def test_parse(self, val, expected):
        assert _parse_port(val) == expected


# ─────────────────────────────────────────────────────────────────────────────
# BeaconingDetector
# ─────────────────────────────────────────────────────────────────────────────


class TestBeaconingDetector:
    def _make_periodic_flows(
        self,
        src: str,
        dst: str,
        interval: float,
        count: int,
        jitter: float = 0.0,
        start: float = 1_000_000.0,
    ):
        """Helper to generate flows at regular intervals."""
        import random

        random.seed(42)
        flows = []
        for i in range(count):
            t = start + i * interval + (random.uniform(-jitter, jitter) if jitter else 0)
            flows.append(make_flow(src_ip=src, dst_ip=dst, timestamp=t))
        return flows

    def test_insufficient_flows_returns_zero(self):
        det = BeaconingDetector(min_flows=5)
        for flow in self._make_periodic_flows("1.1.1.1", "2.2.2.2", 60.0, 3):
            det.update(flow)
        score = det.beaconing_score("1.1.1.1", "2.2.2.2")
        assert score == 0.0

    def test_perfect_periodicity_scores_high(self):
        det = BeaconingDetector(min_flows=5, max_cov=0.3)
        for flow in self._make_periodic_flows("1.1.1.1", "2.2.2.2", 60.0, 20, jitter=0):
            det.update(flow)
        score = det.beaconing_score("1.1.1.1", "2.2.2.2")
        assert score > 0.9

    def test_noisy_traffic_scores_low(self):
        det = BeaconingDetector(min_flows=5, max_cov=0.3)
        # High jitter = irregular = low beaconing score
        for flow in self._make_periodic_flows("1.1.1.1", "2.2.2.2", 60.0, 20, jitter=50.0):
            det.update(flow)
        score = det.beaconing_score("1.1.1.1", "2.2.2.2")
        assert score < 0.5

    def test_score_bounded_zero_to_one(self):
        det = BeaconingDetector(min_flows=5)
        for flow in self._make_periodic_flows("1.1.1.1", "2.2.2.2", 10.0, 30, jitter=100.0):
            det.update(flow)
        score = det.beaconing_score("1.1.1.1", "2.2.2.2")
        assert 0.0 <= score <= 1.0

    def test_unknown_pair_returns_zero(self):
        det = BeaconingDetector()
        assert det.beaconing_score("9.9.9.9", "8.8.8.8") == 0.0

    def test_multiple_pairs_independent(self):
        det = BeaconingDetector(min_flows=5, max_cov=0.3)
        # Pair A: periodic
        for flow in self._make_periodic_flows("A", "B", 60.0, 15, jitter=0):
            det.update(flow)
        # Pair C: random — high jitter
        for flow in self._make_periodic_flows("C", "D", 60.0, 15, jitter=55.0):
            det.update(flow)

        score_ab = det.beaconing_score("A", "B")
        score_cd = det.beaconing_score("C", "D")

        assert score_ab > score_cd
