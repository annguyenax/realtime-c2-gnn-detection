"""Shared pytest fixtures for c2gnn test suite."""

import pytest

from c2gnn.data.flow_builder import FlowRecord


def make_flow(
    src_ip: str = "192.168.1.10",
    dst_ip: str = "10.0.0.1",
    timestamp: float = 1_000_000.0,
    label: str = "normal",
    total_bytes: int = 1024,
    total_fwd_packets: int = 5,
    total_bwd_packets: int = 3,
    duration: float = 1.0,
    protocol: str = "TCP",
    src_port: int = 54321,
    dst_port: int = 80,
) -> FlowRecord:
    """Factory for FlowRecord test instances."""
    safe_dur = max(duration, 1e-9)
    total_pkts = total_fwd_packets + total_bwd_packets
    return FlowRecord(
        timestamp=timestamp,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        duration=duration,
        total_fwd_packets=total_fwd_packets,
        total_bwd_packets=total_bwd_packets,
        total_bytes=total_bytes,
        packet_rate=total_pkts / safe_dur,
        byte_rate=total_bytes / safe_dur,
        flow_iat_mean=0.0,
        flow_iat_std=0.0,
        label=label,
    )


@pytest.fixture
def normal_flow() -> FlowRecord:
    return make_flow(label="normal")


@pytest.fixture
def botnet_flow() -> FlowRecord:
    return make_flow(
        src_ip="192.168.1.99",
        dst_ip="185.220.101.1",
        label="botnet",
        dst_port=6667,
        total_bytes=128,
        duration=0.1,
    )
