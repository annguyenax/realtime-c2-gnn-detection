"""
Download CTU-13 binetflow files for training and generalization testing.

Scenario 10 (Murlo IRC botnet)  → primary training + evaluation
Scenario 8  (Rbot HTTP botnet)  → generalization test (cross-scenario)

Usage:
    python scripts/01_download_ctu13.py
    python scripts/01_download_ctu13.py --scenario 10
    python scripts/01_download_ctu13.py --scenario 8 10
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

# ── Dataset manifest ─────────────────────────────────────────────────────────
# CTU-13 public repository (Czech Technical University in Prague)
SCENARIOS: dict[int, dict] = {
    10: {
        "name": "Murlo IRC C2",
        "url": (
            "https://mcfp.felk.cvut.cz/publicDatasets/"
            "CTU-Malware-Capture-Botnet-10/"
            "capture20110818.pcap.netflow.labeled"
        ),
        "filename": "scenario10.binetflow",
        "approx_mb": 190,
    },
    8: {
        "name": "Rbot HTTP C2",
        "url": (
            "https://mcfp.felk.cvut.cz/publicDatasets/"
            "CTU-Malware-Capture-Botnet-8/"
            "capture20110818-2.pcap.netflow.labeled"
        ),
        "filename": "scenario08.binetflow",
        "approx_mb": 140,
    },
    1: {
        "name": "Neris IRC C2 (small)",
        "url": (
            "https://mcfp.felk.cvut.cz/publicDatasets/"
            "CTU-Malware-Capture-Botnet-1/"
            "capture20110810.pcap.netflow.labeled"
        ),
        "filename": "scenario01.binetflow",
        "approx_mb": 78,
    },
}

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "ctu13"


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        mb_done = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        bar = "#" * int(pct / 2)
        print(
            f"\r  [{bar:<50}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB",
            end="",
            flush=True,
        )
    else:
        mb_done = downloaded / 1_048_576
        print(f"\r  Downloaded {mb_done:.1f} MB ...", end="", flush=True)


def download_scenario(scenario_id: int, force: bool = False) -> Path:
    meta = SCENARIOS[scenario_id]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / meta["filename"]

    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1_048_576
        print(f"  ✓ Already exists: {dest.name} ({size_mb:.1f} MB) — skip")
        return dest

    print(f"\nDownloading Scenario {scenario_id}: {meta['name']}")
    print(f"  URL : {meta['url']}")
    print(f"  Dest: {dest}")
    print(f"  Size: ~{meta['approx_mb']} MB")

    t0 = time.time()
    try:
        urllib.request.urlretrieve(meta["url"], dest, _progress_hook)
    except Exception as exc:
        # Clean up partial file
        if dest.exists():
            dest.unlink()
        print(f"\n  ✗ Download failed: {exc}")
        print(
            "\n  Retry manually:\n"
            f"    curl -L -o {dest} \"{meta['url']}\"\n"
            "  Or download via browser and save to:\n"
            f"    {dest}"
        )
        sys.exit(1)

    elapsed = time.time() - t0
    size_mb = dest.stat().st_size / 1_048_576
    print(f"\n  ✓ Done in {elapsed:.0f}s — {size_mb:.1f} MB")
    return dest


def verify_file(path: Path) -> bool:
    """Quick sanity check: file exists, >10 MB, has CSV header."""
    if not path.exists():
        return False
    if path.stat().st_size < 10_000_000:
        print(f"  ⚠ File too small ({path.stat().st_size} bytes) — may be incomplete")
        return False
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline().strip()
    if "StartTime" not in header or "Label" not in header:
        print(f"  ⚠ Unexpected header: {header[:100]}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CTU-13 binetflow files")
    parser.add_argument(
        "--scenario",
        nargs="+",
        type=int,
        default=[10],
        choices=list(SCENARIOS.keys()),
        help="Scenario IDs to download (default: 10)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file already exists",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CTU-13 Dataset Downloader")
    print(f"Save directory: {RAW_DIR}")
    print("=" * 60)

    downloaded = []
    for sid in args.scenario:
        path = download_scenario(sid, force=args.force)
        ok = verify_file(path)
        if ok:
            print(f"  ✓ Verified: {path.name}")
            downloaded.append(path)
        else:
            print(f"  ✗ Verification failed: {path.name}")

    print("\n" + "=" * 60)
    print(f"Downloaded {len(downloaded)}/{len(args.scenario)} files")
    print("\nNext step:")
    print("  python scripts/02_preprocess.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
