"""
Parse CTU-13 .binetflow files → Polars parquet for fast ML training.

Outputs:
  data/processed/scenario10_train.parquet   (~80%)
  data/processed/scenario10_test.parquet    (~20%)
  data/processed/scenario08_test.parquet    (generalization test)
  data/processed/dataset_stats.json         (label counts, class ratio)

Usage:
    python scripts/02_preprocess.py
    python scripts/02_preprocess.py --scenario 10 --test-size 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from c2gnn.data.flow_builder import CTU13FlowParser, FlowRecord  # noqa: E402

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "ctu13"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def flows_to_dataframe(path: Path, exclude_background: bool = False) -> pl.DataFrame:
    """Parse binetflow file → Polars DataFrame."""
    parser = CTU13FlowParser(exclude_background=exclude_background)
    rows = []

    print(f"  Parsing: {path.name}")
    t0 = time.time()

    for i, flow in enumerate(parser.iter_file(path)):
        rows.append({
            "timestamp": flow.timestamp,
            "src_ip": flow.src_ip,
            "dst_ip": flow.dst_ip,
            "src_port": flow.src_port,
            "dst_port": flow.dst_port,
            "protocol": flow.protocol,
            "duration": flow.duration,
            "total_fwd_packets": flow.total_fwd_packets,
            "total_bwd_packets": flow.total_bwd_packets,
            "total_bytes": flow.total_bytes,
            "packet_rate": flow.packet_rate,
            "byte_rate": flow.byte_rate,
            "flow_iat_mean": flow.flow_iat_mean,
            "flow_iat_std": flow.flow_iat_std,
            "label": flow.label,
        })
        if (i + 1) % 100_000 == 0:
            print(f"    ... {i+1:,} flows parsed ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"  ✓ {len(rows):,} flows in {elapsed:.1f}s")
    return pl.DataFrame(rows)


def print_stats(df: pl.DataFrame, name: str) -> dict:
    """Print label distribution and return stats dict."""
    label_counts = df.group_by("label").len().sort("label")
    total = len(df)

    print(f"\n  {name}:")
    print(f"    Total flows: {total:,}")

    stats = {"total": total, "labels": {}}
    for row in label_counts.iter_rows(named=True):
        lbl = row["label"]
        cnt = row["len"]
        pct = cnt / total * 100
        print(f"    {lbl:<12}: {cnt:>8,}  ({pct:.2f}%)")
        stats["labels"][lbl] = {"count": cnt, "pct": round(pct, 4)}

    botnet_cnt = stats["labels"].get("botnet", {}).get("count", 0)
    stats["botnet_rate"] = botnet_cnt / max(total, 1)
    stats["imbalance_ratio"] = (total - botnet_cnt) / max(botnet_cnt, 1)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess CTU-13 binetflow → parquet")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--exclude-background",
        action="store_true",
        default=False,
        help="Exclude background flows (smaller dataset, higher botnet%)",
    )
    parser.add_argument(
        "--scenario",
        nargs="+",
        type=int,
        default=[10],
        help="Scenarios to process (10 = train/test split, 8 = gen test only)",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    # ── Scenario 10: Train/Test split ────────────────────────────────────────
    if 10 in args.scenario:
        sc10 = RAW_DIR / "scenario10.binetflow"
        if not sc10.exists():
            print(f"✗ Missing: {sc10}")
            print("  Run: python scripts/01_download_ctu13.py --scenario 10")
            sys.exit(1)

        df10 = flows_to_dataframe(sc10, exclude_background=args.exclude_background)
        stats = print_stats(df10, "Scenario 10 (full)")
        all_stats["scenario10_full"] = stats

        # Temporal split: first 80% → train, last 20% → test
        n = len(df10)
        n_train = int(n * (1 - args.test_size))
        df10_sorted = df10.sort("timestamp")
        df_train = df10_sorted[:n_train]
        df_test = df10_sorted[n_train:]

        out_train = PROCESSED_DIR / "scenario10_train.parquet"
        out_test = PROCESSED_DIR / "scenario10_test.parquet"
        df_train.write_parquet(out_train)
        df_test.write_parquet(out_test)

        stats_train = print_stats(df_train, "Scenario 10 TRAIN")
        stats_test = print_stats(df_test, "Scenario 10 TEST")
        all_stats["scenario10_train"] = stats_train
        all_stats["scenario10_test"] = stats_test

        print(f"\n  Saved: {out_train.name} ({out_train.stat().st_size/1e6:.1f} MB)")
        print(f"  Saved: {out_test.name} ({out_test.stat().st_size/1e6:.1f} MB)")

    # ── Scenario 8: Generalization test only ─────────────────────────────────
    if 8 in args.scenario:
        sc08 = RAW_DIR / "scenario08.binetflow"
        if not sc08.exists():
            print(f"\n⚠ Scenario 8 not found: {sc08}")
            print("  To enable generalization test:")
            print("  python scripts/01_download_ctu13.py --scenario 8")
        else:
            df08 = flows_to_dataframe(sc08, exclude_background=args.exclude_background)
            stats08 = print_stats(df08, "Scenario 8 (Rbot, gen test)")
            all_stats["scenario08_test"] = stats08

            out08 = PROCESSED_DIR / "scenario08_test.parquet"
            df08.write_parquet(out08)
            print(f"\n  Saved: {out08.name} ({out08.stat().st_size/1e6:.1f} MB)")

    # ── Save stats JSON ───────────────────────────────────────────────────────
    stats_path = PROCESSED_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Saved stats: {stats_path.name}")

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print("\nNext steps:")
    print("  python scripts/03_train_xgboost.py")
    print("  python scripts/04_train_gnn.py          # requires torch")
    print("=" * 60)


if __name__ == "__main__":
    main()
