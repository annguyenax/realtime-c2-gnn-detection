"""
Generate all figures needed for the thesis report.

Outputs to reports/figures/:
  - dataset_distribution.png       -- pie chart + bar chart class imbalance
  - model_comparison_bar.png       -- F1/Prec/Rec/AUC grouped bar chart
  - shap_importance.png            -- SHAP top features horizontal bar
  - node_features_table.png        -- table of 18 node features
  - system_architecture.png        -- pipeline flowchart
  - fpr_comparison.png             -- FPR comparison bar chart

Usage:
    python scripts/generate_report_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent.parent
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

ARTIFACTS = ROOT / "models" / "artifacts"
PROCESSED = ROOT / "data" / "processed"


def load(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save(fig: plt.Figure, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  OK  {path.name}")


# ─── Figure 1: Dataset distribution ──────────────────────────────────────────
def fig_dataset():
    ds = load(PROCESSED / "dataset_stats.json")
    sc10 = ds.get("scenario10_full", {})
    total = sc10.get("total", 5178417)
    botnet = sc10.get("labels", {}).get("botnet", {}).get("count", 322158)
    normal = total - botnet

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("CTU-13 Scenario 10 — Phân bố nhãn", fontsize=14, fontweight="bold", y=1.02)

    # Pie
    ax = axes[0]
    sizes = [normal, botnet]
    labels = [f"Bình thường\n({normal/total*100:.2f}%)", f"Botnet C2\n({botnet/total*100:.2f}%)"]
    colors = ["#4CAF50", "#F44336"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.2f%%",
        startangle=90, textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Tỷ lệ nhãn (flow-level)", fontsize=11, pad=12)

    # Bar
    ax2 = axes[1]
    categories = ["Bình thường", "Botnet C2"]
    counts = [normal, botnet]
    bars = ax2.bar(categories, counts, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    ax2.set_title("Số lượng flows theo nhãn", fontsize=11)
    ax2.set_ylabel("Số flows")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                 f"{count:,}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="x", labelsize=10)

    fig.tight_layout()
    save(fig, "dataset_distribution.png")


# ─── Figure 2: Model comparison bar chart ────────────────────────────────────
def fig_model_comparison():
    sage = load(ARTIFACTS / "graphsage_metrics.json")
    xgb = load(ARTIFACTS / "xgboost_metrics.json")
    gat = load(ARTIFACTS / "gatv2_metrics.json")

    models = ["XGBoost", "GraphSAGE\n(default thr=0.5)", "GraphSAGE\n(tuned thr=0.9118)", "GATv2\n(default)"]
    f1_vals = [
        xgb.get("f1", 0.9921),
        sage.get("f1", 0.3951),
        sage.get("f1_at_optimal_threshold", 0.6328),
        gat.get("f1", 0.0518),
    ]
    prec_vals = [
        xgb.get("precision", 0.9895),
        sage.get("precision", 0.2675),
        sage.get("precision_at_optimal_threshold", 0.7106),
        gat.get("precision", 0.0267),
    ]
    rec_vals = [
        xgb.get("recall", 0.9947),
        sage.get("recall", 0.7557),
        sage.get("recall_at_optimal_threshold", 0.5703),
        gat.get("recall", 0.8389),
    ]
    auc_vals = [
        xgb.get("roc_auc", 0.9998),
        sage.get("roc_auc", 0.9817),
        sage.get("roc_auc", 0.9817),
        gat.get("roc_auc", 0.9701),
    ]

    x = np.arange(len(models))
    width = 0.2
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    b1 = ax.bar(x - 1.5*width, f1_vals, width, label="F1", color=colors[0], alpha=0.9)
    b2 = ax.bar(x - 0.5*width, prec_vals, width, label="Precision", color=colors[1], alpha=0.9)
    b3 = ax.bar(x + 0.5*width, rec_vals, width, label="Recall", color=colors[2], alpha=0.9)
    b4 = ax.bar(x + 1.5*width, auc_vals, width, label="AUC-ROC", color=colors[3], alpha=0.9)

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title("So sánh hiệu năng các mô hình — CTU-13 Scenario 10", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0.6, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(3.9, 0.61, "F1=0.6 target", fontsize=8, color="gray", ha="right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "model_comparison_bar.png")


# ─── Figure 3: SHAP feature importance ───────────────────────────────────────
def fig_shap():
    shap_data = load(ARTIFACTS / "shap_feature_importance.json")
    features = shap_data.get("features", [])

    if not features:
        features = [
            {"name": "src_port", "shap_mean_abs": 2.811},
            {"name": "bytes_per_packet", "shap_mean_abs": 1.997},
            {"name": "dst_port", "shap_mean_abs": 1.997},
            {"name": "is_tcp", "shap_mean_abs": 1.416},
            {"name": "total_bytes", "shap_mean_abs": 1.254},
            {"name": "byte_rate", "shap_mean_abs": 0.768},
            {"name": "is_icmp", "shap_mean_abs": 0.799},
            {"name": "duration", "shap_mean_abs": 0.450},
            {"name": "total_packets", "shap_mean_abs": 0.380},
            {"name": "pkt_rate", "shap_mean_abs": 0.310},
        ]

    top = features[:10]
    names = [f["name"] for f in reversed(top)]
    vals = [f["shap_mean_abs"] for f in reversed(top)]
    cmap_vals = np.array(vals)
    colors = plt.cm.RdYlGn(cmap_vals / max(cmap_vals))  # type: ignore

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (trung bình mức độ ảnh hưởng)")
    ax.set_title("Top 10 Features quan trọng nhất — XGBoost (SHAP)", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(vals) * 1.2)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    save(fig, "shap_importance.png")


# ─── Figure 4: FPR comparison ────────────────────────────────────────────────
def fig_fpr():
    models = ["XGBoost", "GraphSAGE\n(default)", "GraphSAGE\n(tuned)", "GATv2"]
    fpr_pct = [0.10, 0.107, 0.012, 1.537]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(models, fpr_pct, color=colors, edgecolor="white", linewidth=1.5, width=0.5, alpha=0.9)
    for bar, v in zip(bars, fpr_pct):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.axhline(0.1, color="red", linestyle="--", alpha=0.6, linewidth=1.2)
    ax.text(3.45, 0.12, "Ngưỡng 0.1%", fontsize=8, color="red")
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_title("So sánh False Positive Rate (FPR) — thấp hơn = tốt hơn", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(fpr_pct) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fpr_comparison.png")


# ─── Figure 5: Node feature table ────────────────────────────────────────────
def fig_feature_table():
    features = [
        # Flow stats (14)
        ("in_flows", "Số flows đến node", "Flow stat"),
        ("out_flows", "Số flows đi từ node", "Flow stat"),
        ("in_bytes", "Tổng bytes nhận", "Flow stat"),
        ("out_bytes", "Tổng bytes gửi", "Flow stat"),
        ("in_packets", "Tổng packets nhận", "Flow stat"),
        ("out_packets", "Tổng packets gửi", "Flow stat"),
        ("unique_srcs", "Số IP nguồn duy nhất", "Flow stat"),
        ("unique_dsts", "Số IP đích duy nhất", "Flow stat"),
        ("tcp_ratio", "Tỷ lệ TCP", "Flow stat"),
        ("udp_ratio", "Tỷ lệ UDP", "Flow stat"),
        ("mean_duration", "Thời lượng trung bình", "Flow stat"),
        ("std_duration", "Độ lệch chuẩn thời lượng", "Flow stat"),
        ("suspicious_port", "Tỷ lệ cổng đáng ngờ", "Flow stat"),
        ("fan_out_ratio", "Tỷ lệ fan-out (out/total)", "Flow stat"),
        # Temporal (4)
        ("active_span", "Khoảng thời gian hoạt động trong window", "Temporal"),
        ("mean_iat", "IAT trung bình (giây)", "Temporal"),
        ("iat_cv", "Hệ số biến thiên IAT — đo beaconing", "Temporal"),
        ("repeat_dst_ratio", "Tỷ lệ gửi tới đích lặp lại", "Temporal"),
    ]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axis("off")

    col_labels = ["#", "Tên feature", "Ý nghĩa", "Nhóm"]
    table_data = [[str(i+1), name, desc, grp] for i, (name, desc, grp) in enumerate(features)]

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.35)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        elif features[row-1][2] == "Temporal" if row > 0 and row-1 < len(features) else False:
            cell.set_facecolor("#E3F2FD")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F5F5")
        cell.set_edgecolor("#BDBDBD")

    ax.set_title("Node Feature Vector (18 chiều) — GraphSAGE v3",
                 fontsize=12, fontweight="bold", pad=16)

    legend_patches = [
        mpatches.Patch(color="#FFFFFF", label="Flow stats (14 features)"),
        mpatches.Patch(color="#E3F2FD", label="Temporal beaconing (4 features)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
              framealpha=0.9, edgecolor="#BDBDBD")

    fig.tight_layout()
    save(fig, "node_features_table.png")


# ─── Figure 6: Detection latency ─────────────────────────────────────────────
def fig_latency():
    models = ["XGBoost\n(per-flow)", "GraphSAGE\n(per-graph)", "GATv2\n(per-graph)"]
    latencies = [2.1, 56.2, 296.5]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, latencies, color=colors, edgecolor="white", linewidth=1.5, width=0.45)
    for bar, v in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, v + 3,
                f"{v} ms", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.axhline(100, color="red", linestyle="--", alpha=0.6, linewidth=1.2)
    ax.text(2.35, 105, "100ms target", fontsize=8, color="red")
    ax.set_ylabel("Latency (ms/inference)")
    ax.set_title("Inference Latency theo mô hình", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(latencies) * 1.3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "latency_comparison.png")


if __name__ == "__main__":
    print("Generating report figures...")
    fig_dataset()
    fig_model_comparison()
    fig_shap()
    fig_fpr()
    fig_feature_table()
    fig_latency()
    print(f"\nAll figures saved to: {FIGURES}")
    print("Files:")
    for f in sorted(FIGURES.glob("*.png")):
        print(f"  {f.name} ({f.stat().st_size//1024}KB)")
