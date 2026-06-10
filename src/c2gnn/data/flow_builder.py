"""
flow_builder.py — Thread 1: Flow Builder

Parse CTU-13 .binetflow (NetFlow) files thành FlowRecord chuẩn hóa.
Hỗ trợ realtime replay theo timestamp.

Author: Member 1 (Data/Network/Security Engineer)
"""

from __future__ import annotations

import collections
import csv
import math
import queue
import threading
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FlowRecord:
    """
    Normalized, immutable flow record — single unit of network communication.

    Designed to be the canonical format across all pipeline stages.
    All fields are sanitized and typed at parse time.
    """

    timestamp: float  # Unix epoch seconds (float)
    src_ip: str
    dst_ip: str
    src_port: int  # 0 if not applicable
    dst_port: int  # 0 if not applicable
    protocol: str  # "TCP" | "UDP" | "ICMP" | "OTHER"
    duration: float  # seconds
    total_fwd_packets: int
    total_bwd_packets: int
    total_bytes: int
    packet_rate: float  # packets/second
    byte_rate: float  # bytes/second
    flow_iat_mean: float  # inter-arrival time mean (0 if not available)
    flow_iat_std: float  # inter-arrival time std  (0 if not available)
    label: str  # "normal" | "botnet" | "background"

    # Derived convenience properties
    @property
    def is_malicious(self) -> bool:
        return self.label == "botnet"

    @property
    def is_background(self) -> bool:
        return self.label == "background"

    @property
    def total_packets(self) -> int:
        return self.total_fwd_packets + self.total_bwd_packets

    @property
    def bytes_per_packet(self) -> float:
        return self.total_bytes / max(self.total_packets, 1)

    @property
    def dst_port_is_well_known(self) -> bool:
        return 0 < self.dst_port < 1024

    def to_feature_dict(self) -> dict[str, float]:
        """Serialize to flat dict for XGBoost / sklearn ingestion."""
        return {
            "duration": self.duration,
            "total_fwd_packets": float(self.total_fwd_packets),
            "total_bwd_packets": float(self.total_bwd_packets),
            "total_packets": float(self.total_packets),
            "total_bytes": float(self.total_bytes),
            "packet_rate": self.packet_rate,
            "byte_rate": self.byte_rate,
            "flow_iat_mean": self.flow_iat_mean,
            "flow_iat_std": self.flow_iat_std,
            "src_port": float(self.src_port),
            "dst_port": float(self.dst_port),
            "bytes_per_packet": self.bytes_per_packet,
            "fwd_bwd_ratio": self.total_fwd_packets / max(self.total_bwd_packets, 1),
            "dst_port_well_known": float(self.dst_port_is_well_known),
            "is_tcp": float(self.protocol == "TCP"),
            "is_udp": float(self.protocol == "UDP"),
            "is_icmp": float(self.protocol == "ICMP"),
            "label_binary": float(self.is_malicious),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CTU-13 Parser
# ─────────────────────────────────────────────────────────────────────────────

# CTU-13 binetflow column reference:
# StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport,
# State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label

_CTU13_LABEL_BOTNET_KEYWORDS = frozenset(
    [
        "botnet",
        "from-botnet",
        "to-botnet",
        "from-normal",
        "to-normal",
    ]
)

_PROTOCOL_NORMALIZE: dict[str, str] = {
    "tcp": "TCP",
    "udp": "UDP",
    "icmp": "ICMP",
    "icmp6": "ICMP",
    "igmp": "OTHER",
    "arp": "OTHER",
    "ipv6": "OTHER",
}


def _normalize_label(raw: str) -> str:
    """
    CTU-13 labels are inconsistent across scenarios.
    We simplify to 3 classes: botnet / normal / background.

    Examples of raw labels:
      "flow=Background"
      "flow=Normal"
      "flow=From-Botnet-V42-UDP-DNS"
      "flow=To-Botnet-V42-TCP-CC5"
    """
    if not raw:
        return "background"

    lower = raw.lower().strip()

    # Check for background first (most flows are background)
    if "background" in lower:
        return "background"

    # Botnet: any flow that has "botnet" in label
    if "botnet" in lower:
        return "botnet"

    if "normal" in lower:
        return "normal"

    return "background"


def _parse_timestamp(ts_str: str) -> float:
    """Parse CTU-13 timestamp string to Unix epoch float.

    Handles two CTU-13 timestamp formats:
      - Slash:  "2011/08/10 09:46:53.047925"
      - Dash:   "2011-08-18 10:19:13.328"
    """
    if not ts_str:
        return time.time()

    try:
        dot_idx = ts_str.rfind(".")
        if dot_idx != -1:
            dt_part = ts_str[:dot_idx]
            frac = float("0" + ts_str[dot_idx:])
        else:
            dt_part = ts_str
            frac = 0.0

        for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(dt_part, fmt).timestamp() + frac
            except ValueError:
                continue
    except Exception:
        pass

    try:
        return float(ts_str)
    except ValueError:
        logger.warning("Cannot parse timestamp", ts_str=ts_str)
        return time.time()


def _split_ip_port(combined: str) -> tuple[str, int]:
    """Split 'IP:port' into (ip, port). Handles missing port and hex ports."""
    combined = combined.strip()
    if ":" not in combined:
        return combined, 0
    # Last colon separates host from port (handles IPv6 like ::1:80 poorly, but CTU-13 is IPv4)
    host, _, port_str = combined.rpartition(":")
    return host or combined, _parse_port(port_str)


def _parse_port(port_val: object) -> int:
    """Handle CTU-13 port values: decimal, hex (0x...), or empty."""
    s = str(port_val).strip()
    if not s:
        return 0
    try:
        if s.startswith("0x") or s.startswith("0X"):
            return int(s, 16)
        return int(float(s))  # handles "80.0" etc.
    except (ValueError, OverflowError):
        return 0


class CTU13FlowParser:
    """
    Parser for CTU-13 .binetflow (NetFlow v5) labeled files.

    Usage:
        parser = CTU13FlowParser()
        for flow in parser.iter_file(Path("capture.binetflow")):
            print(flow)
    """

    def __init__(
        self,
        exclude_background: bool = False,
        min_duration: float = 0.0,
    ):
        """
        Args:
            exclude_background: If True, skip background flows (reduces dataset 10x).
            min_duration: Skip flows shorter than this (seconds).
        """
        self.exclude_background = exclude_background
        self.min_duration = min_duration
        self._stats: dict[str, int] = collections.Counter()

    def iter_file(self, filepath: Path) -> Generator[FlowRecord, None, None]:
        """
        Yield FlowRecord objects from a CTU-13 .binetflow/.netflow.labeled file.
        Auto-detects two CTU-13 formats:
          - CSV binetflow: StartTime,Dur,Proto,SrcAddr,Sport,...
          - Argus labeled: tab-separated "Date flow start  Durat  Prot  Src IP Addr:Port  ->"
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, newline="", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()

        argus = "Src IP Addr" in first_line or "Date flow start" in first_line

        with open(filepath, newline="", encoding="utf-8", errors="replace") as f:
            if argus:
                f.readline()  # skip header
                yield from self._iter_argus(f)
            else:
                reader = csv.DictReader(f, skipinitialspace=True)
                for row in reader:
                    record = self._parse_row(row)
                    if record is None:
                        self._stats["skipped"] += 1
                        continue
                    if self.exclude_background and record.is_background:
                        self._stats["excluded_background"] += 1
                        continue
                    if record.duration < self.min_duration:
                        self._stats["excluded_short"] += 1
                        continue
                    self._stats[f"label_{record.label}"] += 1
                    self._stats["total"] += 1
                    yield record

        logger.info("Parsing complete", **self._stats)

    def _iter_argus(self, f) -> Generator[FlowRecord, None, None]:
        """
        Parse argus/nfdump tab-separated format used by CTU-13 netflow.labeled files.

        The CTU-13 argus format has two variants depending on protocol:
          Variant A (TCP/UDP, port included in IP field):
            ts | dur | proto | IP:port | -> | IP:port | flags | tos | pkts | bytes | flows | label
          Variant B (ICMP/PIM/other, port as separate empty column):
            ts | dur | proto | IP | (empty) | -> | IP | (empty) | flags | tos | pkts | bytes | flows | label

        Strategy: locate the '->' direction field, then read relative to it.
        """
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split("\t")
            if len(fields) < 12:
                self._stats["skipped"] += 1
                continue

            try:
                # Find '->' direction indicator (robust to field count variation)
                try:
                    dir_idx = next(i for i, v in enumerate(fields) if v.strip() in ("->", "<->", "<-"))
                except StopIteration:
                    self._stats["skipped"] += 1
                    continue

                # Fields before direction: [ts, dur, proto, src_ip_or_ipport, (optional empty)]
                # Fields after  direction: [dst_ip_or_ipport, (optional empty), flags, tos, pkts, bytes, flows, label]
                src_raw = fields[dir_idx - 1].strip() or (fields[dir_idx - 2].strip() if dir_idx >= 2 else "")
                dst_raw = fields[dir_idx + 1].strip() or (fields[dir_idx + 2].strip() if dir_idx + 2 < len(fields) else "")

                src_ip, src_port = _split_ip_port(src_raw)
                dst_ip, dst_port = _split_ip_port(dst_raw)

                if not src_ip or not dst_ip or src_ip == "0.0.0.0":
                    self._stats["skipped"] += 1
                    continue

                # After dst field(s), remaining: flags, tos, packets, bytes, flows, label
                # Step past dst and optional empty port column
                after_dst = dir_idx + 2
                if after_dst < len(fields) and not fields[after_dst].strip():
                    after_dst += 1  # skip empty port column

                # after_dst now points to flags; +1=tos, +2=pkts, +3=bytes, +4=flows, +5=label
                if after_dst + 5 >= len(fields):
                    self._stats["skipped"] += 1
                    continue

                duration = float(fields[1] or 0)
                tot_pkts = max(0, int(float(fields[after_dst + 2] or 0)))
                tot_bytes = max(0, int(float(fields[after_dst + 3] or 0)))
                label_raw = fields[after_dst + 5].strip()

                # Some files have a 13th detailed label at after_dst+6 — prefer it if present
                if after_dst + 6 < len(fields) and fields[after_dst + 6].strip():
                    label_raw = fields[after_dst + 6].strip()

                safe_dur = max(duration, 1e-9)
                protocol = _PROTOCOL_NORMALIZE.get(fields[2].lower().strip(), "OTHER")
                label = _normalize_label(label_raw)

                record = FlowRecord(
                    timestamp=_parse_timestamp(fields[0]),
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port,
                    protocol=protocol,
                    duration=duration,
                    total_fwd_packets=tot_pkts // 2 + tot_pkts % 2,
                    total_bwd_packets=tot_pkts // 2,
                    total_bytes=tot_bytes,
                    packet_rate=tot_pkts / safe_dur,
                    byte_rate=tot_bytes / safe_dur,
                    flow_iat_mean=0.0,
                    flow_iat_std=0.0,
                    label=label,
                )
            except (ValueError, TypeError, IndexError) as exc:
                logger.debug("Argus row parse error", error=str(exc))
                self._stats["skipped"] += 1
                continue

            if self.exclude_background and record.is_background:
                self._stats["excluded_background"] += 1
                continue
            if record.duration < self.min_duration:
                self._stats["excluded_short"] += 1
                continue

            self._stats[f"label_{record.label}"] += 1
            self._stats["total"] += 1
            yield record

    def parse_to_dataframe(self, filepath: Path) -> pl.DataFrame:
        """
        Parse entire file to Polars DataFrame for batch ML processing.
        Uses Polars for ~5x faster processing than pandas.
        """
        records = list(self.iter_file(filepath))
        if not records:
            return pl.DataFrame()

        rows = [r.to_feature_dict() for r in records]
        df = pl.from_dicts(rows)

        # Add metadata columns
        meta = pl.from_dicts(
            [
                {
                    "timestamp": r.timestamp,
                    "src_ip": r.src_ip,
                    "dst_ip": r.dst_ip,
                    "protocol": r.protocol,
                    "label": r.label,
                }
                for r in records
            ]
        )

        return pl.concat([meta, df.drop(["label_binary"])], how="horizontal")

    def _parse_row(self, row: dict[str, str]) -> FlowRecord | None:
        """Convert one CSV row to FlowRecord. Returns None on error."""
        try:
            src_ip = row.get("SrcAddr", "").strip()
            dst_ip = row.get("DstAddr", "").strip()

            # Skip if IPs are missing or clearly invalid
            if not src_ip or not dst_ip or src_ip == "0.0.0.0":
                return None

            duration = float(row.get("Dur", 0) or 0)
            tot_pkts = max(0, int(float(row.get("TotPkts", 0) or 0)))
            tot_bytes = max(0, int(float(row.get("TotBytes", 0) or 0)))
            src_bytes = max(0, int(float(row.get("SrcBytes", 0) or 0)))

            # Estimate fwd/bwd split from SrcBytes
            fwd_pkts = max(0, int(tot_pkts * src_bytes / max(tot_bytes, 1)))
            bwd_pkts = max(0, tot_pkts - fwd_pkts)

            safe_dur = max(duration, 1e-9)
            packet_rate = tot_pkts / safe_dur
            byte_rate = tot_bytes / safe_dur

            proto_raw = row.get("Proto", "").lower().strip()
            protocol = _PROTOCOL_NORMALIZE.get(proto_raw, "OTHER")

            return FlowRecord(
                timestamp=_parse_timestamp(row.get("StartTime", "")),
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=_parse_port(row.get("Sport", 0)),
                dst_port=_parse_port(row.get("Dport", 0)),
                protocol=protocol,
                duration=duration,
                total_fwd_packets=fwd_pkts,
                total_bwd_packets=bwd_pkts,
                total_bytes=tot_bytes,
                packet_rate=packet_rate,
                byte_rate=byte_rate,
                flow_iat_mean=0.0,  # binetflow doesn't have IAT; use PCAP for this
                flow_iat_std=0.0,
                label=_normalize_label(row.get("Label", "")),
            )

        except (ValueError, TypeError, KeyError) as exc:
            logger.debug("Row parse error", error=str(exc))
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Beaconing Detector (C2 behavior analysis)
# ─────────────────────────────────────────────────────────────────────────────


class BeaconingDetector:
    """
    Heuristic beaconing detector per (src_ip, dst_ip) pair.

    C2 beaconing hallmarks:
      - Periodic inter-arrival times (low CoV)
      - Low volume per connection
      - Long-lived presence in traffic
    """

    def __init__(self, min_flows: int = 5, max_cov: float = 0.3):
        self.min_flows = min_flows
        self.max_cov = max_cov  # Coefficient of Variation threshold
        self._arrival_times: dict[tuple[str, str], list[float]] = collections.defaultdict(list)

    def update(self, flow: FlowRecord) -> None:
        key = (flow.src_ip, flow.dst_ip)
        self._arrival_times[key].append(flow.timestamp)

    def beaconing_score(self, src_ip: str, dst_ip: str) -> float:
        """
        Returns [0, 1] beaconing probability.
        High score = periodic = likely C2 beacon.
        """
        arrivals = sorted(self._arrival_times.get((src_ip, dst_ip), []))
        if len(arrivals) < self.min_flows:
            return 0.0

        # Inter-arrival times
        iats = [arrivals[i + 1] - arrivals[i] for i in range(len(arrivals) - 1)]
        mean_iat = sum(iats) / len(iats)

        if mean_iat < 1e-6:
            return 0.0

        std_iat = math.sqrt(sum((x - mean_iat) ** 2 for x in iats) / len(iats))
        cov = std_iat / mean_iat  # Coefficient of Variation

        # Low CoV = periodic = beaconing
        # Map CoV to [0, 1]: cov=0 → score=1, cov=max_cov → score=0
        score = max(0.0, 1.0 - cov / self.max_cov)
        return round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Thread 1 — Flow Builder Worker
# ─────────────────────────────────────────────────────────────────────────────


class FlowBuilderWorker(threading.Thread):
    """
    Thread 1 of the realtime pipeline.

    Reads flow data (simulating realtime) and pushes FlowRecord objects
    to the downstream flow_queue.

    Flow:
        CTU-13 file → replay by timestamp → flow_queue

    Args:
        data_path: Path to .binetflow or .parquet file
        output_queue: Destination queue (consumed by GraphUpdateWorker)
        realtime_factor: >1 speeds up playback. 0 = send as fast as possible.
        max_flows: Optional cap (useful for testing)
        stop_event: Set to stop the thread early
    """

    def __init__(
        self,
        data_path: Path,
        output_queue: queue.Queue,
        realtime_factor: float = 10.0,
        max_flows: int | None = None,
        stop_event: threading.Event | None = None,
        exclude_background: bool = True,
    ):
        super().__init__(name="FlowBuilder", daemon=True)
        self.data_path = data_path
        self.output_queue = output_queue
        self.realtime_factor = realtime_factor
        self.max_flows = max_flows
        self.stop_event = stop_event or threading.Event()
        self.exclude_background = exclude_background

        self._flows_sent = 0
        self._skipped = 0
        self._start_wall: float = 0.0

    def run(self) -> None:
        log = logger.bind(thread="FlowBuilder", path=str(self.data_path))
        log.info("Starting")
        self._start_wall = time.time()

        flows: Iterator[FlowRecord]

        # Support both .binetflow (CSV) and .parquet
        if self.data_path.suffix in (".binetflow", ".csv", ".labeled"):
            parser = CTU13FlowParser(exclude_background=self.exclude_background)
            flows = parser.iter_file(self.data_path)
        elif self.data_path.suffix == ".parquet":
            flows = self._iter_parquet()
        else:
            log.error("Unsupported file format", suffix=self.data_path.suffix)
            self.output_queue.put(None)
            return

        prev_ts: float | None = None

        for flow in flows:
            if self.stop_event.is_set():
                break
            if self.max_flows and self._flows_sent >= self.max_flows:
                break

            # Realtime simulation: sleep proportional to time gap
            if prev_ts is not None and self.realtime_factor > 0:
                gap = (flow.timestamp - prev_ts) / self.realtime_factor
                # Cap sleep to 2 seconds to avoid stalling
                if 0 < gap < 2.0:
                    time.sleep(gap)

            try:
                self.output_queue.put(flow, timeout=5.0)
                self._flows_sent += 1
                prev_ts = flow.timestamp

                if self._flows_sent % 5_000 == 0:
                    elapsed = time.time() - self._start_wall
                    log.info(
                        "Progress",
                        flows_sent=self._flows_sent,
                        elapsed_s=round(elapsed, 1),
                        throughput_fps=round(self._flows_sent / max(elapsed, 1), 0),
                    )
            except queue.Full:
                log.warning("Flow queue full, dropping flow")
                self._skipped += 1

        # Send sentinel to signal downstream workers
        self.output_queue.put(None)
        elapsed = time.time() - self._start_wall
        log.info(
            "Done",
            flows_sent=self._flows_sent,
            skipped=self._skipped,
            total_elapsed_s=round(elapsed, 1),
        )

    def _iter_parquet(self) -> Generator[FlowRecord, None, None]:
        """Read pre-processed parquet and reconstruct FlowRecords."""
        df = pl.read_parquet(self.data_path).sort("timestamp")

        for row in df.iter_rows(named=True):
            yield FlowRecord(
                timestamp=float(row["timestamp"]),
                src_ip=str(row["src_ip"]),
                dst_ip=str(row["dst_ip"]),
                src_port=int(row.get("src_port", 0)),
                dst_port=int(row.get("dst_port", 0)),
                protocol=str(row.get("protocol", "OTHER")),
                duration=float(row.get("duration", 0)),
                total_fwd_packets=int(row.get("total_fwd_packets", 0)),
                total_bwd_packets=int(row.get("total_bwd_packets", 0)),
                total_bytes=int(row.get("total_bytes", 0)),
                packet_rate=float(row.get("packet_rate", 0)),
                byte_rate=float(row.get("byte_rate", 0)),
                flow_iat_mean=float(row.get("flow_iat_mean", 0)),
                flow_iat_std=float(row.get("flow_iat_std", 0)),
                label=str(row.get("label", "background")),
            )

    @property
    def stats(self) -> dict:
        return {
            "flows_sent": self._flows_sent,
            "skipped": self._skipped,
            "elapsed_s": round(time.time() - self._start_wall, 1),
        }
