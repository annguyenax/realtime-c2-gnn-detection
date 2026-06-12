"""
src/c2gnn/dashboard/app.py
Realtime C2 Botnet Detection — SOC Dashboard

Demo mode: ground truth labels (CTU-13) shown alongside predictions
for evaluation. Uses @st.fragment for flicker-free auto-refresh.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_METRICS_PATH = PROJECT_ROOT / "reports" / "final_metrics.json"

RISK_HIGH = 0.50
RISK_MED  = 0.35

COLOR_HIGH   = "#FF4136"
COLOR_MED    = "#FF851B"
COLOR_TP     = "#2ECC40"
COLOR_FP     = "#FF4136"
COLOR_ACCENT = "#0074D9"

# ──────────────────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────────────────


def _get(path: str, params: dict | None = None) -> Any:
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def fetch_alerts(limit: int = 200, min_score: float = 0.0) -> list[dict]:
    return _get("/api/v1/alerts", {"limit": limit, "min_score": min_score}) or []


def fetch_stats() -> dict[str, Any]:
    return _get("/api/v1/stats") or {}


def fetch_botnet_ips() -> list[str]:
    return _get("/api/v1/botnet_ips") or []


def fetch_health() -> bool:
    d = _get("/api/v1/health")
    return bool(d and d.get("status") == "ok")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def score_label(s: float) -> str:
    if s >= RISK_HIGH: return "🔴 HIGH"
    if s >= RISK_MED:  return "🟠 MEDIUM"
    return "⚪ LOW"


def gt_label(is_known: bool | None) -> str:
    if is_known is True:  return "✅ BOTNET"
    if is_known is False: return "⬜ BENIGN"
    return "❓"


def status_label(is_known: bool | None) -> str:
    if is_known is True:  return "🎯 TRUE POSITIVE"
    if is_known is False: return "❌ FALSE POSITIVE"
    return "—"


def alerts_to_df(alerts: list[dict]) -> pd.DataFrame:
    rows = []
    for a in alerts:
        p = a.get("payload", {})
        ik = p.get("is_known_botnet")
        rows.append({
            "ID": a.get("alert_id"),
            "Timestamp": p.get("timestamp", "")[:19].replace("T", " "),
            "Src IP": p.get("src_ip"),
            "Dst IP": p.get("dst_ip"),
            "Risk Score": round(p.get("risk_score", 0), 4),
            "Severity": score_label(p.get("risk_score", 0)),
            "Ground Truth": gt_label(ik),
            "Status": status_label(ik),
            "Reasons": " | ".join(p.get("reason", [])),
            "_ik": ik,
        })
    cols = ["ID","Timestamp","Src IP","Dst IP","Risk Score","Severity",
            "Ground Truth","Status","Reasons","_ik"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


# ──────────────────────────────────────────────────────────────────────────────
# Page shell (static — renders once, no blink)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="C2 GNN Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stDataFrame { font-size:13px; }
/* Suppress Streamlit's iframe loading flash */
iframe { background: #0E1117 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar — static config
with st.sidebar:
    st.title("⚙️ Config")
    refresh_s = st.slider("Refresh (s)", 2, 30, 3)
    min_score_filter = st.slider("Min risk score", 0.0, 1.0, 0.0, step=0.05)
    max_alerts = int(st.number_input("Max alerts", 50, 500, 200))
    st.divider()
    st.caption("CTU-13 Scenario 10 — real botnet captures")
    st.caption("GraphSAGE on dynamic IP–IP graph")
    st.caption("Ground truth: for verification only, not used in inference")

st.title("🔍 Realtime C2 Botnet Detection — CTU-13 Replay")

# ──────────────────────────────────────────────────────────────────────────────
# Live section — wrapped in @st.fragment → refreshes WITHOUT page blink
# ──────────────────────────────────────────────────────────────────────────────


@st.fragment(run_every=refresh_s)
def live_dashboard() -> None:
    stats = fetch_stats()
    alerts_raw = fetch_alerts(limit=max_alerts, min_score=min_score_filter)
    df = alerts_to_df(alerts_raw)
    known_ips = fetch_botnet_ips()
    known_set = set(known_ips)
    api_ok = fetch_health()

    tp = stats.get("tp_count", 0)
    fp = stats.get("fp_count", 0)
    total_tagged = tp + fp
    precision = stats.get("precision", 0.0)

    # ── API status + timestamp ───────────────────────────────────────────────
    status_icon = "🟢 Connected" if api_ok else "🔴 Disconnected"
    st.caption(
        f"⏱ {datetime.now().strftime('%H:%M:%S')}  |  API: {status_icon}  |  "
        "Ground truth labels for demo evaluation only"
    )

    # ── Stats header ─────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Alerts", stats.get("total_alerts", 0),
              delta=f"+{stats.get('alerts_last_60s', 0)} /60s")
    c2.metric("🔴 HIGH Risk", stats.get("high_risk_alerts", 0))
    c3.metric("🎯 True Positives", tp)
    c4.metric("❌ False Positives", fp)
    c5.metric("📊 Precision", f"{precision:.1%}" if total_tagged > 0 else "—")
    c6.metric("🦠 Known Botnets", stats.get("known_botnet_count", len(known_ips)))

    # ── Banner ───────────────────────────────────────────────────────────────
    if tp > 0:
        st.success(
            f"🎯 **{tp} BOTNET HOST{'S' if tp > 1 else ''} DETECTED** — "
            "GNN identified known C2/botnet traffic from CTU-13 dataset"
        )
    elif stats.get("total_alerts", 0) > 0:
        st.warning("⚠️ Alerts active — botnet IPs scoring near threshold, waiting for dense phase")
    else:
        st.info("⏳ Building IP graph... alerts appear after ~300s of simulated traffic")

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_feed, tab_verify, tab_timeline, tab_graph, tab_ips, tab_models = st.tabs([
        "📋 Alert Feed",
        "🦠 Botnet Verification",
        "📈 Timeline",
        "🕸️ Graph View",
        "🎯 Top IPs",
        "📊 Models",
    ])

    # Tab 1 — Alert Feed
    with tab_feed:
        st.subheader("Live Alert Feed")
        st.caption("🟢 row = confirmed botnet (TP)  |  🔴 row = false alarm (FP)")
        if df.empty:
            st.info("No alerts yet — graph is warming up...")
        else:
            display = df.drop(columns=["_ik"])

            def row_style(row):
                ik = df.loc[row.name, "_ik"] if "_ik" in df.columns else None
                if ik is True:
                    return ["background-color:#2ECC4018; color:#2ECC40; font-weight:600"] * len(row)
                if ik is False:
                    return ["background-color:#FF413618"] * len(row)
                return [""] * len(row)

            def score_style(v: float) -> str:
                if v >= RISK_HIGH: return f"color:{COLOR_HIGH};font-weight:700"
                if v >= RISK_MED:  return f"color:{COLOR_MED}"
                return ""

            styled = (
                display.style
                .apply(row_style, axis=1)
                .map(score_style, subset=["Risk Score"])
                .format({"Risk Score": "{:.4f}"})
            )
            st.dataframe(styled, use_container_width=True, height=480)
            st.download_button(
                "⬇️ Download CSV",
                display.to_csv(index=False),
                file_name=f"c2gnn_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    # Tab 2 — Botnet Verification
    with tab_verify:
        st.subheader("🦠 Ground Truth Verification")
        st.caption(
            "CTU-13 labels loaded from dataset — **not** used during inference. "
            "Shown here to validate model accuracy against known ground truth."
        )
        if not known_ips:
            st.info("Known botnet IPs will be registered when pipeline starts.")
        else:
            detected: dict[str, float] = {}
            for a in alerts_raw:
                p = a.get("payload", {})
                if p.get("is_known_botnet") is True:
                    ip = p.get("src_ip", "")
                    detected[ip] = max(detected.get(ip, 0), p.get("risk_score", 0))

            missed = [ip for ip in known_ips if ip not in detected]
            recall = len(detected) / max(len(known_ips), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Known Botnet IPs", len(known_ips))
            m2.metric("Detected ✅", len(detected), delta=f"{recall:.0%} recall")
            m3.metric("Missed ⏳", len(missed))
            m4.metric("F1 Score", f"{f1:.3f}")

            if detected:
                st.markdown("#### ✅ Detected Botnet Hosts")
                det_df = pd.DataFrame([
                    {"IP": ip, "Max Risk Score": round(s, 4), "Result": "🎯 DETECTED"}
                    for ip, s in sorted(detected.items())
                ])
                st.dataframe(
                    det_df.style.map(
                        lambda v: "color:#2ECC40;font-weight:700" if v == "🎯 DETECTED" else "",
                        subset=["Result"],
                    ),
                    use_container_width=True,
                )

            if missed:
                with st.expander(f"⏳ Not yet detected ({len(missed)} IPs)"):
                    st.caption("May appear in later phases or score just below threshold.")
                    st.dataframe(pd.DataFrame({"IP": sorted(missed)}), use_container_width=True)

            st.info(
                "**Note**: Ground truth includes C2 servers, infected hosts, AND external IPs "
                "contacted by bots (e.g., Google servers scanned by botnet). "
                "High FPR on external IPs is expected — they exhibit similar fan-out patterns."
            )

    # Tab 3 — Timeline
    with tab_timeline:
        st.subheader("Detection Timeline — TP vs FP")
        if df.empty:
            st.info("No data yet.")
        else:
            try:
                df_t = df.copy()
                df_t["minute"] = pd.to_datetime(df_t["Timestamp"]).dt.floor("1min")
                df_t["Verdict"] = df_t["_ik"].map(
                    {True: "🎯 TRUE POSITIVE", False: "❌ FALSE POSITIVE", None: "⚪ UNKNOWN"}
                ).fillna("⚪ UNKNOWN")
                tl = df_t.groupby(["minute", "Verdict"]).size().reset_index(name="count")
                fig = px.bar(
                    tl, x="minute", y="count", color="Verdict",
                    color_discrete_map={
                        "🎯 TRUE POSITIVE": COLOR_TP,
                        "❌ FALSE POSITIVE": COLOR_FP,
                        "⚪ UNKNOWN": "#777",
                    },
                    title="Alerts per Minute",
                    labels={"minute": "Time", "count": "Count"},
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

            fig2 = px.histogram(
                df, x="Risk Score", nbins=40,
                color="_ik",
                color_discrete_map={True: COLOR_TP, False: COLOR_FP, None: "#777"},
                title="Risk Score Distribution (green=TP, red=FP)",
                template="plotly_dark",
            )
            fig2.add_vline(x=RISK_MED, line_dash="dash", line_color=COLOR_MED, annotation_text="MEDIUM")
            fig2.add_vline(x=RISK_HIGH, line_dash="dash", line_color=COLOR_HIGH, annotation_text="HIGH")
            st.plotly_chart(fig2, use_container_width=True)

    # Tab 4 — Graph
    with tab_graph:
        st.subheader("IP Communication Graph")
        st.caption("🟢 = confirmed botnet  🔴 = false alarm  🔵 = unknown")
        plot_df = df[df["Risk Score"] >= RISK_MED].head(200) if not df.empty else df
        if plot_df.empty:
            st.info("No alerts above MEDIUM threshold yet.")
        else:
            G = nx.DiGraph()
            node_scores: dict[str, float] = {}
            node_ik: dict[str, bool | None] = {}
            for _, row in plot_df.iterrows():
                src, dst, ik = row["Src IP"], row["Dst IP"], row["_ik"]
                if src and dst:
                    G.add_edge(src, dst)
                    node_scores[src] = max(node_scores.get(src, 0), row["Risk Score"])
                    if ik is not None:
                        node_ik[src] = ik

            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, seed=42, k=2.0)
                ex, ey = [], []
                for u, v in G.edges():
                    x0, y0 = pos[u]; x1, y1 = pos[v]
                    ex += [x0, x1, None]; ey += [y0, y1, None]
                et = go.Scatter(x=ex, y=ey, line={"width": 0.8, "color": "#555"},
                                hoverinfo="none", mode="lines")
                colors = [COLOR_TP if node_ik.get(n) is True
                          else COLOR_FP if node_ik.get(n) is False
                          else COLOR_ACCENT for n in G.nodes()]
                texts = [f"<b>{n}</b><br>Score:{node_scores.get(n,0):.3f}<br>"
                         f"{'✅ BOTNET' if node_ik.get(n) is True else '❌ FP' if node_ik.get(n) is False else '❓'}"
                         for n in G.nodes()]
                nt = go.Scatter(x=[pos[n][0] for n in G.nodes()],
                                y=[pos[n][1] for n in G.nodes()],
                                mode="markers", hoverinfo="text", text=texts,
                                marker={"size": [8 + G.degree(n)*3 for n in G.nodes()],
                                        "color": colors, "line": {"width": 1.5, "color": "#222"}})
                fig_g = go.Figure(data=[et, nt], layout=go.Layout(
                    title=f"IP Graph — {G.number_of_nodes()} nodes",
                    showlegend=False, hovermode="closest", template="plotly_dark",
                    xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                    yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                    height=580,
                ))
                st.plotly_chart(fig_g, use_container_width=True)

    # Tab 5 — Top IPs
    with tab_ips:
        st.subheader("Top Suspicious Source IPs")
        top_ips_data = stats.get("top_src_ips", [])
        if not top_ips_data:
            st.info("No data yet.")
        else:
            top_df = pd.DataFrame(top_ips_data, columns=["IP Address", "Alert Count", "Max Risk Score"])
            top_df["Ground Truth"] = top_df["IP Address"].apply(
                lambda ip: "✅ BOTNET" if ip in known_set else "⬜ BENIGN"
            )
            top_df = top_df.sort_values("Alert Count", ascending=False)
            fig_top = px.bar(
                top_df.head(15), x="IP Address", y="Alert Count",
                color="Max Risk Score", color_continuous_scale="RdYlGn_r",
                title="Top 15 Source IPs", template="plotly_dark",
            )
            st.plotly_chart(fig_top, use_container_width=True)
            st.dataframe(top_df, use_container_width=True)

    # Tab 6 — Models
    with tab_models:
        st.subheader("Model Performance — CTU-13 Scenario 10")
        if FINAL_METRICS_PATH.exists():
            with open(FINAL_METRICS_PATH, encoding="utf-8") as f:
                fm = json.load(f)
            rows = [
                {"Model": m, **{k: v for k, v in met.items()
                                if k in ["precision","recall","f1","roc_auc","pr_auc"]}}
                for m, met in fm.get("models", {}).items()
            ]
            if rows:
                rdf = pd.DataFrame(rows)
                st.dataframe(rdf.set_index("Model"), use_container_width=True)
                cats = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
                fig_r = go.Figure()
                for i, row in rdf.iterrows():
                    vals = [row.get(c, 0) for c in cats] + [row.get(cats[0], 0)]
                    fig_r.add_trace(go.Scatterpolar(
                        r=vals, theta=[c.upper() for c in cats] + [cats[0].upper()],
                        fill="toself", name=row["Model"], opacity=0.6,
                    ))
                fig_r.update_layout(
                    polar={"radialaxis": {"visible": True, "range": [0, 1]}},
                    showlegend=True, title="Model Comparison", template="plotly_dark",
                )
                st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.warning("No metrics file found. Run training first.")

        if stats.get("model_distribution"):
            md = stats["model_distribution"]
            fig_pie = px.pie(names=list(md.keys()), values=list(md.values()),
                             title="Live Alerts by Model", template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)


# ── Call the live fragment (runs every refresh_s seconds, no page blink) ──────
live_dashboard()
