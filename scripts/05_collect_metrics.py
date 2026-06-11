"""
Collect all training results into a single JSON for report.

Reads:
  models/artifacts/xgboost_metrics.json
  models/artifacts/graphsage_metrics.json   (if exists)
  models/artifacts/gatv2_metrics.json        (if exists)
  models/artifacts/shap_feature_importance.json
  data/processed/dataset_stats.json

Writes:
  reports/final_metrics.json   ← paste into report
  reports/results_table.txt    ← copy-paste comparison table

Usage:
    python scripts/05_collect_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent.parent / "models" / "artifacts"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load all metrics ──────────────────────────────────────────────────────
    xgb = load_json(ARTIFACTS_DIR / "xgboost_metrics.json")
    sage = load_json(ARTIFACTS_DIR / "graphsage_metrics.json")
    gat = load_json(ARTIFACTS_DIR / "gatv2_metrics.json")
    shap = load_json(ARTIFACTS_DIR / "shap_feature_importance.json")
    dataset = load_json(PROCESSED_DIR / "dataset_stats.json")

    if not xgb:
        print("FAIL No metrics found. Run training first:")
        print("  python scripts/03_train_xgboost.py")
        sys.exit(1)

    # ── Build unified report ──────────────────────────────────────────────────
    report = {
        "dataset": {
            "name": "CTU-13 Botnet Dataset",
            "scenario_train": "Scenario 10 (Murlo IRC C2)",
            "scenario_gen_test": "Scenario 8 (Rbot HTTP C2)",
        },
        "models": {},
    }

    # XGBoost
    report["models"]["xgboost"] = {
        "type": "Tabular ML (no graph)",
        "precision": round(xgb.get("precision", 0), 4),
        "recall": round(xgb.get("recall", 0), 4),
        "f1": round(xgb.get("f1", 0), 4),
        "roc_auc": round(xgb.get("roc_auc", 0), 4),
        "pr_auc": round(xgb.get("pr_auc", 0), 4),
        "false_positive_rate_pct": round(xgb.get("false_positive_rate", 0) * 100, 2),
        "latency_mean_ms": round(xgb.get("latency_mean_ms", 0), 2),
        "cv_f1_mean": round(xgb.get("cv_f1_mean", 0), 4),
        "cv_f1_std": round(xgb.get("cv_f1_std", 0), 4),
    }
    if xgb.get("gen_f1"):
        report["models"]["xgboost"]["gen_f1_scenario8"] = round(xgb["gen_f1"], 4)

    # GraphSAGE — report both default (thr=0.5) and val-tuned threshold
    if sage:
        report["models"]["graphsage"] = {
            "type": "GNN - GraphSAGE (3 layers, dim=128)",
            # Default threshold=0.5
            "precision_default": round(sage.get("precision", 0), 4),
            "recall_default": round(sage.get("recall", 0), 4),
            "f1_default": round(sage.get("f1", 0), 4),
            "false_positive_rate_pct_default": round(sage.get("false_positive_rate", 0) * 100, 3),
            # Val-tuned threshold (primary result for report)
            "optimal_threshold": round(sage.get("optimal_threshold", 0.5), 4),
            "precision_tuned": round(sage.get("precision_at_optimal_threshold", 0), 4),
            "recall_tuned": round(sage.get("recall_at_optimal_threshold", 0), 4),
            "f1_tuned": round(sage.get("f1_at_optimal_threshold", 0), 4),
            "false_positive_rate_pct_tuned": round(sage.get("fpr_at_optimal_threshold", 0) * 100, 3),
            # Model-level (threshold-independent)
            "roc_auc": round(sage.get("roc_auc", 0), 4),
            "pr_auc": round(sage.get("pr_auc", 0), 4),
            "latency_mean_ms": round(sage.get("latency_mean_ms", 0), 2),
        }

    # GATv2
    if gat:
        report["models"]["gatv2"] = {
            "type": "GNN - GATv2 (2 layers, 4 heads, dim=64)",
            "precision_default": round(gat.get("precision", 0), 4),
            "recall_default": round(gat.get("recall", 0), 4),
            "f1_default": round(gat.get("f1", 0), 4),
            "false_positive_rate_pct_default": round(gat.get("false_positive_rate", 0) * 100, 3),
            "optimal_threshold": round(gat.get("optimal_threshold", 0.5), 4),
            "precision_tuned": round(gat.get("precision_at_optimal_threshold", gat.get("precision", 0)), 4),
            "recall_tuned": round(gat.get("recall_at_optimal_threshold", gat.get("recall", 0)), 4),
            "f1_tuned": round(gat.get("f1_at_optimal_threshold", gat.get("f1", 0)), 4),
            "false_positive_rate_pct_tuned": round(gat.get("fpr_at_optimal_threshold", gat.get("false_positive_rate", 0)) * 100, 3),
            "roc_auc": round(gat.get("roc_auc", 0), 4),
            "pr_auc": round(gat.get("pr_auc", 0), 4),
            "latency_mean_ms": round(gat.get("latency_mean_ms", 0), 2),
        }

    # SHAP top features
    if shap and shap.get("features"):
        report["top_features_shap"] = shap["features"][:10]

    # Dataset stats
    if dataset:
        sc10 = dataset.get("scenario10_full", {})
        report["dataset"]["total_flows"] = sc10.get("total", 0)
        report["dataset"]["botnet_rate_pct"] = round(sc10.get("botnet_rate", 0) * 100, 2)
        report["dataset"]["imbalance_ratio"] = round(sc10.get("imbalance_ratio", 0), 1)
        report["dataset"]["label_counts"] = sc10.get("labels", {})

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = REPORTS_DIR / "final_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"OK Saved: {out_json}")

    # ── Print comparison table ─────────────────────────────────────────────────
    lines = []
    lines.append("=" * 88)
    lines.append("MODEL COMPARISON - CTU-13 SCENARIO 10")
    lines.append("=" * 88)
    lines.append(f"{'Model':<20} {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8} {'FPR%':>7} {'ms/g':>7}")
    lines.append("-" * 88)

    for name, m in report["models"].items():
        if name == "xgboost":
            lines.append(
                f"{name:<20} {'0.5 (default)':>10} "
                f"{m['precision']:>10.4f} "
                f"{m['recall']:>8.4f} "
                f"{m['f1']:>8.4f} "
                f"{m['roc_auc']:>8.4f} "
                f"{m['false_positive_rate_pct']:>6.2f}% "
                f"{m['latency_mean_ms']:>6.1f}"
            )
        else:
            thr_default = "0.5 (default)"
            thr_tuned = f"{m.get('optimal_threshold', 0.5):.4f} (tuned)"
            lines.append(
                f"{name:<20} {thr_default:>10} "
                f"{m.get('precision_default', 0):>10.4f} "
                f"{m.get('recall_default', 0):>8.4f} "
                f"{m.get('f1_default', 0):>8.4f} "
                f"{m.get('roc_auc', 0):>8.4f} "
                f"{m.get('false_positive_rate_pct_default', 0):>6.3f}% "
                f"{m.get('latency_mean_ms', 0):>6.1f}"
            )
            lines.append(
                f"{name:<20} {thr_tuned:>10} "
                f"{m.get('precision_tuned', 0):>10.4f} "
                f"{m.get('recall_tuned', 0):>8.4f} "
                f"{m.get('f1_tuned', 0):>8.4f} "
                f"{'':>8} "
                f"{m.get('false_positive_rate_pct_tuned', 0):>6.3f}% "
                f"{'':>6}"
            )
            lines.append("")
    lines.append("=" * 88)

    table_text = "\n".join(lines)
    print("\n" + table_text)

    table_path = REPORTS_DIR / "results_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table_text + "\n")
    print(f"\nOK Saved: {table_path}")

    if report["top_features_shap"]:
        print("\nTop SHAP features (XGBoost):")
        for feat in report["top_features_shap"][:7]:
            print(f"  {feat['name']:<25} : {feat['shap_mean_abs']:.4f}")

    print("\n" + "=" * 75)
    print("Copy numbers above into your report.")
    print("Full details in: reports/final_metrics.json")
    print("=" * 75)


if __name__ == "__main__":
    main()
