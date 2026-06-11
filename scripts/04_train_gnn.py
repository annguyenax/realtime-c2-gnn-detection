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
    print(f"FAIL Missing dependency: {e}")
    print("\nInstall with:")
    print("  pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu")
    print("  pip install torch-geometric pyg-lib torch-scatter torch-sparse \\")
    print("    -f https://data.pyg.org/whl/torch-2.3.0+cpu.html")
    sys.exit(1)

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch_geometric.data import Data

from c2gnn.data.flow_builder import CTU13FlowParser
from c2gnn.graph.dynamic_graph import SlidingWindowGraph
from c2gnn.models.graphsage import GATv2C2Detector, GNNTrainer, GraphSAGEC2Detector

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

    print(f"  OK {flow_count:,} flows -> {len(snapshots)} graph snapshots")
    return snapshots


def evaluate_on_snapshots(
    model: torch.nn.Module,
    snapshots: list[Data],
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Run inference on all snapshots, collect node-level predictions."""
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for data in snapshots:
            data = data.to(device)
            if not hasattr(data, "y") or data.y is None:
                continue
            edge_attr = data.edge_attr
            if edge_attr is not None and edge_attr.shape[1] > 7:
                edge_attr = edge_attr[:, :7]
            out = model(data.x, data.edge_index, edge_attr)
            probs = torch.softmax(out, dim=-1)[:, 1]  # P(botnet)
            preds = (probs > threshold).long()
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
        "pr_auc": float(average_precision_score(y_true, y_score)),
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
            edge_attr = data.edge_attr
            if edge_attr is not None and edge_attr.shape[1] > 7:
                edge_attr = edge_attr[:, :7]
            _ = model(data.x, data.edge_index, edge_attr)

    for data in samples:
        data = data.to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            edge_attr = data.edge_attr
            if edge_attr is not None and edge_attr.shape[1] > 7:
                edge_attr = edge_attr[:, :7]
            _ = model(data.x, data.edge_index, edge_attr)
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
    max_class_weight: float = 50.0,
    filter_empty_snapshots: bool = True,
    use_focal_loss: bool = False,
    focal_gamma: float = 1.5,
    min_recall: float = 0.40,
    epochs: int = 100,
    patience: int = 12,
    hidden_channels: int = 128,
    dropout: float = 0.3,
) -> dict[str, float]:
    from c2gnn.models.graphsage import NODE_FEATURE_DIM as _NODE_DIM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model — in_channels auto-reads NODE_FEATURE_DIM so it stays in sync
    if model_type == "graphsage":
        model = GraphSAGEC2Detector(
            in_channels=_NODE_DIM,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=3,
            dropout=dropout,
        ).to(device)
    elif model_type == "gatv2":
        model = GATv2C2Detector(
            in_channels=_NODE_DIM,
            hidden_channels=hidden_channels,
            out_channels=2,
            heads=4,
            dropout=dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    trainer = GNNTrainer(model, device=str(device))

    loss_desc = f"FocalLoss(gamma={focal_gamma})" if use_focal_loss else f"WeightedCE(cap={max_class_weight})"
    print(f"\n  Training {model_type.upper()} [{loss_desc}] on {len(train_snapshots)} snapshots...")
    print(f"  node_feature_dim={_NODE_DIM}, hidden={hidden_channels}, dropout={dropout}")
    print(f"  epochs={epochs}, patience={patience}, filter_empty={filter_empty_snapshots}")
    print(f"  Evaluating on {len(test_snapshots)} snapshots...")

    model_path = ARTIFACTS_DIR / f"{model_type}_best.pt"
    train_result = trainer.train(
        train_graphs=train_snapshots,
        val_graphs=test_snapshots[: max(1, min(50, len(test_snapshots)))],
        epochs=epochs,
        patience=patience,
        save_path=model_path,
        run_name=f"{model_type}-ctu13-s10",
        max_class_weight=max_class_weight,
        filter_empty_snapshots=filter_empty_snapshots,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        tune_threshold=True,
        min_recall=min_recall,
    )
    optimal_threshold = train_result.get("optimal_threshold", 0.5)
    print(f"\n  Optimal threshold (from val): {optimal_threshold:.4f}")

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))

    # Full evaluation at both threshold=0.5 and optimal threshold
    print("\n  Final evaluation on test set...")
    metrics_default = evaluate_on_snapshots(model, test_snapshots, device, threshold=0.5)
    metrics_tuned   = evaluate_on_snapshots(model, test_snapshots, device, threshold=optimal_threshold)
    latency = measure_inference_latency_gnn(model, test_snapshots, device)

    print(f"  --- threshold=0.5 (default) ---")
    print(f"  F1  : {metrics_default.get('f1', 0):.4f}  Prec: {metrics_default.get('precision', 0):.4f}  Rec: {metrics_default.get('recall', 0):.4f}")
    print(f"  AUC : {metrics_default.get('roc_auc', 0):.4f}  FPR: {metrics_default.get('false_positive_rate', 0)*100:.2f}%")
    print(f"  --- threshold={optimal_threshold:.4f} (val-tuned) ---")
    print(f"  F1  : {metrics_tuned.get('f1', 0):.4f}  Prec: {metrics_tuned.get('precision', 0):.4f}  Rec: {metrics_tuned.get('recall', 0):.4f}")
    print(f"  FPR : {metrics_tuned.get('false_positive_rate', 0)*100:.2f}%")
    print(f"  Lat : {latency.get('latency_mean_ms', 0):.1f} ms/graph")

    print(f"  Saved: {model_path.name}")

    # Save metrics at default threshold (reproducible baseline) plus tuned threshold info
    all_metrics = {
        **metrics_default,
        **latency,
        "optimal_threshold": round(optimal_threshold, 6),
        "f1_at_optimal_threshold": round(metrics_tuned.get("f1", 0.0), 6),
        "precision_at_optimal_threshold": round(metrics_tuned.get("precision", 0.0), 6),
        "recall_at_optimal_threshold": round(metrics_tuned.get("recall", 0.0), 6),
        "fpr_at_optimal_threshold": round(metrics_tuned.get("false_positive_rate", 0.0), 6),
    }
    metrics_path = ARTIFACTS_DIR / f"{model_type}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: round(float(v), 6) for k, v in all_metrics.items()}, f, indent=2)
    print(f"  Saved metrics: {metrics_path.name}")

    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GNN models (GraphSAGE / GATv2)")
    parser.add_argument("--model", choices=["graphsage", "gatv2", "all"], default="all")
    parser.add_argument("--window-size", type=float, default=300.0, help="Graph window in seconds")
    parser.add_argument(
        "--max-class-weight",
        type=float,
        default=50.0,
        help="Cap for botnet class weight (raw n_neg/n_pos can be >1000 with small windows, "
             "causing precision collapse). Default: 50",
    )
    parser.add_argument(
        "--filter-empty",
        action="store_true",
        default=True,
        help="Remove all-normal snapshots from training set (default: on)",
    )
    parser.add_argument(
        "--no-filter-empty",
        dest="filter_empty",
        action="store_false",
        help="Keep all-normal snapshots during training",
    )
    parser.add_argument(
        "--focal-loss",
        action="store_true",
        default=False,
        help="Use FocalLoss instead of weighted CrossEntropy. "
             "When set, class weight is NOT applied (avoids double reweighting). "
             "Start with --focal-gamma 1.5.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.5,
        help="Focal loss gamma parameter (default: 1.5). "
             "Higher values focus more on hard examples. Try: 1.0, 1.5, 2.0",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.40,
        help="Minimum recall floor for threshold tuning (default: 0.40). "
             "Thresholds that drop recall below this are skipped.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100). CosineAnnealingLR T_max "
             "is set to this value so the full LR schedule is exercised.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=12,
        help="Early stopping patience — epochs without val_f1 improvement (default: 12).",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=128,
        help="Hidden layer width for GNN (default: 128). Try 256 for larger model.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate (default: 0.3). Try 0.2 if model underfits.",
    )
    args = parser.parse_args()

    sc10_path = RAW_DIR / "scenario10.binetflow"
    if not sc10_path.exists():
        print("FAIL CTU-13 Scenario 10 not found.")
        print("  Run: python scripts/01_download_ctu13.py")
        sys.exit(1)

    print("=" * 60)
    print("GNN Training (GraphSAGE / GATv2)")
    print("=" * 60)

    snapshots = build_graph_dataset(sc10_path, window_size=args.window_size)

    if len(snapshots) < 10:
        print(f"FAIL Only {len(snapshots)} snapshots - not enough to train. Try smaller window.")
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
        result = train_model(
            model_type,
            train_snaps,
            test_snaps,
            max_class_weight=args.max_class_weight,
            filter_empty_snapshots=args.filter_empty,
            use_focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma,
            min_recall=args.min_recall,
            epochs=args.epochs,
            patience=args.patience,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
        )
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
