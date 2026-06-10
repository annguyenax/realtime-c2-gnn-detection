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
import ssl
import sys
import time
import urllib.request
from pathlib import Path

# Windows Python 3.13 may not have root CA for mcfp.felk.cvut.cz
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# ── Dataset manifest ─────────────────────────────────────────────────────────
# CTU-13 public repository (Czech Technical University in Prague)
SCENARIOS: dict[int, dict] = {
    10: {
        # CTU paper Scenario 10 = internal Botnet-51, Murlo IRC C2
        "name": "Murlo IRC C2 (Scenario 10)",
        "url": (
            "https://mcfp.felk.cvut.cz/publicDatasets/"
            "CTU-Malware-Capture-Botnet-51/"
            "capture20110818.pcap.netflow.labeled"
        ),
        "filename": "scenario10.binetflow",
        "approx_mb": 190,
    },
    8: {
        # CTU paper Scenario 8 = internal Botnet-49, Menti botnet
        "name": "Menti Botnet (Scenario 8)",
        "url": (
            "https://mcfp.felk.cvut.cz/publicDatasets/"
            "CTU-Malware-Capture-Botnet-49/"
            "capture20110816-3.pcap.netflow.labeled"
        ),
        "filename": "scenario08.binetflow",
        "approx_mb": 60,
    },
    1: {
        # CTU paper Scenario 1 = internal Botnet-42, Neris IRC C2
        "name": "Neris IRC C2 (Scenario 1)",
        "url": (
            "https://mcfp.felk.cvut.cz/publicDatasets/"
            "CTU-Malware-Capture-Botnet-42/"
            "capture20110810.pcap.netflow.labeled"
        ),
        "filename": "scenario01.binetflow",
        "approx_mb": 340,
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
        print(f"  OK Already exists: {dest.name} ({size_mb:.1f} MB) -- skip")
        return dest

    print(f"\nDownloading Scenario {scenario_id}: {meta['name']}")
    print(f"  URL : {meta['url']}")
    print(f"  Dest: {dest}")
    print(f"  Size: ~{meta['approx_mb']} MB")

    t0 = time.time()
    try:
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=_SSL_CTX)
        )
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(meta["url"], dest, _progress_hook)
    except Exception as exc:
        # Clean up partial file
        if dest.exists():
            dest.unlink()
        print(f"\n  FAILED: {exc}")
        print(
            "\n  Retry manually:\n"
            f"    curl -k -L -o {dest} \"{meta['url']}\"\n"
            "  Or download via browser and save to:\n"
            f"    {dest}"
        )
        sys.exit(1)

    elapsed = time.time() - t0
    size_mb = dest.stat().st_size / 1_048_576
    print(f"\n  OK Done in {elapsed:.0f}s -- {size_mb:.1f} MB")
    return dest


def verify_file(path: Path) -> bool:
    """Quick sanity check: file exists, >10 MB, has CSV header."""
    if not path.exists():
        return False
    if path.stat().st_size < 10_000_000:
        print(f"  WARN File too small ({path.stat().st_size} bytes) -- may be incomplete")
        return False
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline().strip()
    if "StartTime" not in header or "Label" not in header:
        print(f"  WARN Unexpected header: {header[:100]}")
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
            print(f"  OK Verified: {path.name}")
            downloaded.append(path)
        else:
            print(f"  FAIL Verification failed: {path.name}")

    print("\n" + "=" * 60)
    print(f"Downloaded {len(downloaded)}/{len(args.scenario)} files")
    print("\nNext step:")
    print("  python scripts/02_preprocess.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
