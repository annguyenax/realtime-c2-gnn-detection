"""
Train GraphSAGE and GATv2 on CTU-13 Scenario 10.

Requires: torch, torch-geometric
  pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
  pip install torch-geometric

Outputs:
  models/artifacts/graphsage_best.pt
  models/artifacts/gatv2_best.pt
  models/artifacts/graphsage_metrics.json
  models/artifacts/gatv2_metrics.json

Usage:
    python scripts/04_train_gnn.py --model graphsage
    python scripts/04_train_gnn.py --model gatv2
    python scripts/04_train_gnn.py --model all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Check torch availability
try:
    import torch
    import torch_geometric  # noqa: F401
    print(f"  PyTorch   : {torch.__version__}")
    print(f"  PyG       : {torch_geometric.__version__}")
    print(f"  Device    : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("\nInstall with:")
    print("  pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu")
    print("  pip install torch-geometric pyg-lib torch-scatter torch-sparse \\")
    print("    -f https://data.pyg.org/whl/torch-2.3.0+cpu.html")
    sys.exit(1)

import numpy as np
import polars as pl
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from c2gnn.graph.dynamic_graph import SlidingWindowGraph
from c2gnn.data.flow_builder import CTU13FlowParser
from c2gnn.models.graphsage import GATv2Model, GNNTrainer, GraphSAGEModel

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "ctu13"
ARTIFACTS_DIR = Path(__file__).parent.parent / "models" / "artifacts"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def build_graph_dataset(binetflow_path: Path, window_size: float = 300.0) -> list[Data]:
    """
    Build PyG graph snapshots from binetflow using sliding window.
    Each snapshot = one training sample.
    """
    print(f"  Building graph snapshots from {binetflow_path.name}...")
    parser = CTU13FlowParser(exclude_background=False)
    graph = SlidingWindowGraph(window_size=window_size, edge_ttl=window_size * 2)

    snapshots: list[Data] = []
    snapshot_interval = window_size / 2  # 50% overlap
    last_snapshot_time = None
    flow_count = 0

    for flow in parser.iter_file(binetflow_path):
        graph.update(flow)
        flow_count += 1

        if last_snapshot_time is None:
            last_snapshot_time = flow.timestamp

        if flow.timestamp - last_snapshot_time >= snapshot_interval:
            data = graph.to_pyg_data(include_ground_truth=True)
            if data is not None and data.x.shape[0] >= 3:
                snapshots.append(data)
            last_snapshot_time = flow.timestamp

        if flow_count % 50_000 == 0:
            print(f"    ... {flow_count:,} flows, {len(snapshots)} snapshots", flush=True)

    print(f"  ✓ {flow_count:,} flows → {len(snapshots)} graph snapshots")
    return snapshots


def evaluate_on_snapshots(
    model: torch.nn.Module,
    snapshots: list[Data],
    device: torch.device,
) -> dict[str, float]:
    """Run inference on all snapshots, collect node-level predictions."""
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for data in snapshots:
            data = data.to(device)
            if not hasattr(data, "y") or data.y is None:
                continue
            out = model(data.x, data.edge_index)
            probs = torch.softmax(out, dim=-1)[:, 1]  # P(botnet)
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(data.y.cpu().numpy().tolist())
            all_scores.extend(probs.cpu().numpy().tolist())

    if not all_labels:
        return {}

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_score = np.array(all_scores)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = fp / max(tn + fp, 1)

    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "false_positive_rate": float(fpr),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
    }


def measure_inference_latency_gnn(
    model: torch.nn.Module,
    snapshots: list[Data],
    device: torch.device,
    n_samples: int = 50,
) -> dict[str, float]:
    model.eval()
    latencies = []
    samples = snapshots[:n_samples]

    # Warmup
    for data in samples[:5]:
        data = data.to(device)
        with torch.no_grad():
            _ = model(data.x, data.edge_index)

    for data in samples:
        data = data.to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(data.x, data.edge_index)
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "latency_mean_ms": float(arr.mean()),
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
    }


def train_model(
    model_type: str,
    train_snapshots: list[Data],
    test_snapshots: list[Data],
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    if model_type == "graphsage":
        model = GraphSAGEModel(
            in_channels=14,
            hidden_channels=128,
            out_channels=2,
            num_layers=3,
            dropout=0.3,
        ).to(device)
    elif model_type == "gatv2":
        model = GATv2Model(
            in_channels=14,
            hidden_channels=64,
            out_channels=2,
            heads=4,
            dropout=0.3,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    trainer = GNNTrainer(model, device=device)

    print(f"\n  Training {model_type.upper()} on {len(train_snapshots)} snapshots...")
    print(f"  Evaluating on {len(test_snapshots)} snapshots...")

    # Train
    best_val_f1 = 0.0
    best_state = None
    epochs = 50

    for epoch in range(1, epochs + 1):
        # Mini-epoch: one pass over all training snapshots
        total_loss = 0.0
        for data in train_snapshots:
            loss = trainer.train_step(data)
            total_loss += loss

        if epoch % 10 == 0:
            metrics = evaluate_on_snapshots(model, test_snapshots[:50], device)
            val_f1 = metrics.get("f1", 0.0)
            print(f"  Epoch {epoch:3d} | loss={total_loss:.4f} | val_f1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Full evaluation
    print(f"\n  Final evaluation on test set...")
    metrics = evaluate_on_snapshots(model, test_snapshots, device)
    latency = measure_inference_latency_gnn(model, test_snapshots, device)

    print(f"  F1  : {metrics.get('f1', 0):.4f}")
    print(f"  AUC : {metrics.get('roc_auc', 0):.4f}")
    print(f"  FPR : {metrics.get('false_positive_rate', 0)*100:.2f}%")
    print(f"  Lat : {latency.get('latency_mean_ms', 0):.1f} ms/graph")

    # Save model
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / f"{model_type}_best.pt"
    torch.save({"model_state": model.state_dict(), "model_type": model_type}, model_path)
    print(f"  Saved: {model_path.name}")

    all_metrics = {**metrics, **latency}
    metrics_path = ARTIFACTS_DIR / f"{model_type}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: round(float(v), 6) for k, v in all_metrics.items()}, f, indent=2)
    print(f"  Saved metrics: {metrics_path.name}")

    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GNN models (GraphSAGE / GATv2)")
    parser.add_argument("--model", choices=["graphsage", "gatv2", "all"], default="all")
    parser.add_argument("--window-size", type=float, default=300.0, help="Graph window in seconds")
    args = parser.parse_args()

    sc10_path = RAW_DIR / "scenario10.binetflow"
    if not sc10_path.exists():
        print("✗ CTU-13 Scenario 10 not found.")
        print("  Run: python scripts/01_download_ctu13.py")
        sys.exit(1)

    print("=" * 60)
    print("GNN Training (GraphSAGE / GATv2)")
    print("=" * 60)

    snapshots = build_graph_dataset(sc10_path, window_size=args.window_size)

    if len(snapshots) < 10:
        print(f"✗ Only {len(snapshots)} snapshots — not enough to train. Try smaller window.")
        sys.exit(1)

    # 80/20 temporal split
    n_train = int(len(snapshots) * 0.8)
    train_snaps = snapshots[:n_train]
    test_snaps = snapshots[n_train:]
    print(f"\n  Train snapshots: {len(train_snaps)}")
    print(f"  Test  snapshots: {len(test_snaps)}")

    models_to_train = ["graphsage", "gatv2"] if args.model == "all" else [args.model]
    all_results = {}

    for model_type in models_to_train:
        print(f"\n{'='*40}")
        print(f"Training: {model_type.upper()}")
        print(f"{'='*40}")
        result = train_model(model_type, train_snaps, test_snaps)
        all_results[model_type] = result

    # Summary
    print("\n" + "=" * 60)
    print("GNN RESULTS SUMMARY (for report):")
    print(f"{'Model':<12} {'F1':>8} {'AUC':>8} {'FPR':>8} {'Latency ms':>12}")
    print("-" * 60)
    for name, m in all_results.items():
        print(
            f"{name:<12} "
            f"{m.get('f1', 0):>8.4f} "
            f"{m.get('roc_auc', 0):>8.4f} "
            f"{m.get('false_positive_rate', 0)*100:>7.2f}% "
            f"{m.get('latency_mean_ms', 0):>10.1f} ms"
        )
    print("=" * 60)
    print("\nNext step:")
    print("  python scripts/05_collect_metrics.py")


if __name__ == "__main__":
    main()
