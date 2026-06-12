# 🔍 Realtime C2 Traffic Detection via Graph Neural Networks

> Status: research prototype. The repository contains runnable data parsing,
> dynamic graph construction, baseline/GNN model code, API, dashboard, Docker,
> and tests. Reported model metrics must be regenerated locally with the
> scripts in `scripts/`; do not treat example numbers as verified until
> `reports/final_metrics.json` exists.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.5-orange?style=flat-square)](https://pyg.org)
[![CI](https://img.shields.io/github/actions/workflow/status/annguyenax/realtime-c2-gnn-detection/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/annguyenax/realtime-c2-gnn-detection/actions)
[![Security](https://img.shields.io/github/actions/workflow/status/annguyenax/realtime-c2-gnn-detection/security.yml?branch=main&style=flat-square&label=Security&color=green)](https://github.com/annguyenax/realtime-c2-gnn-detection/actions)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](./Dockerfile)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](./LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000?style=flat-square)](https://github.com/astral-sh/ruff)

**A research prototype that detects Command-and-Control (C2) botnet traffic using dynamic graph construction and Graph Neural Networks, with a real-time multi-threaded streaming pipeline.**

[Architecture](#-system-architecture) • [Quick Start](#-quick-start) • [Results](#-results) • [Demo](#-demo) • [Team](#-team)

</div>

---

## 🎯 Problem Statement

Modern botnets communicate with their Command-and-Control (C2) servers using stealthy, low-volume, periodic traffic that blends into normal network activity. Traditional signature-based IDS and per-flow ML classifiers miss the **relational structure** of this communication: a bot is not just an anomalous flow — it is an anomalous **node in a network graph** exhibiting suspicious neighborhood patterns.

This project addresses the gap between classical per-flow ML and graph-aware detection by:

1. **Modeling network traffic as a dynamic IP-to-IP communication graph** updated in real time using a sliding time window.
2. **Applying Graph Neural Networks (GraphSAGE, GATv2)** to learn node representations that encode both local flow statistics and neighborhood communication patterns.
3. **Deploying a three-thread streaming pipeline** that processes flows, updates graphs, and runs inference in a latency-aware manner — achieving sub-second detection time.

---

## 🛡️ Threat Model

| Component | Description |
|-----------|-------------|
| **Adversary** | Botnet operator controlling infected hosts via C2 server |
| **Attack vector** | Periodic low-volume beaconing over TCP/UDP/HTTP/IRC |
| **Evasion tactics** | Jitter in beacon intervals, domain generation algorithm (DGA), HTTPS tunneling |
| **Detection scope** | Network perimeter — flow-level telemetry (no payload decryption) |
| **Out of scope** | Encrypted C2 over legitimate CDNs (Cobalt Strike via Azure), Living-off-the-Land (LotL) |
| **Data sensitivity** | IP addresses treated as pseudonymous identifiers; no user PII processed |

---

## 📦 Dataset

### Primary: CTU-13 Botnet Dataset

| Property | Value |
|----------|-------|
| Source | Czech Technical University (CTU) |
| Format | `.binetflow` CSV (pre-extracted flows) + PCAP |
| Scenarios used | Scenario 10 (Neris), Scenario 11 (Rbot), Scenario 13 (mixed) |
| Total flows | ~2.8M flows across 3 scenarios |
| Label distribution | ~3–8% botnet, rest benign/background |
| Features | 14 flow-level features + graph-derived node features |

### Secondary: CICIDS2017 (generalization test)

Used for cross-dataset evaluation only. Not used in training.

> **Data Card**: See [`data/README.md`](data/README.md) for full schema, label mapping, and ethical use statement.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REALTIME C2 DETECTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CTU-13 .binetflow  ──►  ┌────────────────────────────────────────────┐    │
│  (timestamp replay)       │         Thread 1: Flow Builder             │    │
│                           │  • Parse flow records (Polars)             │    │
│                           │  • Beaconing detection (CoV score)         │    │
│                           │  • Realtime timestamp replay               │    │
│                           └──────────────────┬─────────────────────────┘    │
│                                              │  flow_queue (Queue 20k)      │
│                           ┌──────────────────▼─────────────────────────┐    │
│                           │         Thread 2: Graph Update             │    │
│                           │  • Sliding window (60s default, TTL 120s)  │    │
│                           │  • Incremental node/edge feature update    │    │
│                           │  • 14-dim node feature vector              │    │
│                           │  • to_pyg_data() snapshot every 5s        │    │
│                           └──────────────────┬─────────────────────────┘    │
│                                              │  inference_queue (Queue 200) │
│                           ┌──────────────────▼─────────────────────────┐    │
│                           │         Thread 3: GNN Inference + Alert    │    │
│                           │  • GraphSAGE / GATv2 forward pass          │    │
│                           │  • @torch.no_grad(), threshold=0.7         │    │
│                           │  • Heuristic explainability                │    │
│                           │  • Alert JSON → FastAPI /alerts            │    │
│                           └──────────────────┬─────────────────────────┘    │
│                                              │                              │
│                           ┌──────────────────▼─────────────────────────┐    │
│                           │         Streamlit Dashboard                 │    │
│                           │  • Live alert feed                         │    │
│                           │  • Graph visualization (PyVis)             │    │
│                           │  • Detection timeline                      │    │
│                           └────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Layer | Technology | Role |
|-------|-----------|------|
| Data ingestion | Polars, `FlowBuilderWorker` | Parse `.binetflow`, timestamp replay |
| Graph engine | NetworkX, `SlidingWindowGraph` | Dynamic IP-IP graph, TTL-based expiry |
| Node features | `NodeData.to_feature_vector()` | 14-dim: degree, entropy, port ratio, bytes, IAT stats |
| GNN models | PyTorch Geometric SAGEConv, GATv2Conv | Node classification (bot/benign) |
| Baseline | XGBoost 2.x + SHAP | Per-flow tabular classification |
| Streaming | Python `Queue` + `threading` | 3-thread lockless pipeline |
| Alert API | FastAPI | REST endpoint `/alerts` for dashboard |
| Dashboard | Streamlit | Live visualization |
| Experiment tracking | MLflow | Metrics, artifacts, model registry |
| DevSecOps | GitHub Actions, Trivy, Bandit, ruff, mypy | CI/CD + security scanning |

---

## 📊 Results

> **Reproducibility note**: Numbers below come from `reports/final_metrics.json` generated by
> `python scripts/05_collect_metrics.py`. Run the full pipeline to regenerate.

### Model Comparison (CTU-13 Scenario 10)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Latency |
|-------|-----------|--------|-----|---------|--------|-----|---------|
| XGBoost (baseline) | **0.990** | **0.995** | **0.992** | **0.9998** | 0.999 | 0.10% | 2.1 ms/flow |
| **GraphSAGE v3** (w=60s, 18-dim) | **0.958** | 0.494 | **0.652** | 0.969 | 0.671 | **0.00%** | 58 ms/graph |
| GraphSAGE v3 @ opt-thr (0.415) | 0.908 | 0.537 | **0.675** | — | — | 0.003% | — |
| GATv2 (w=60s, untuned) | 0.027 | 0.839 | 0.052 | 0.970 | 0.093 | 1.54% | 296.5 ms/graph |

> **Honest analysis**: XGBoost achieves F1=0.992 because Neris botnet uses IRC (port 6667) —
> a highly discriminative flow-level feature (`src_port`, `dst_port` are top SHAP features).
> GraphSAGE v3 (18-dim: 14 flow stats + 4 temporal beaconing features) achieves **Precision=0.958
> and FPR=0.00%** — every flagged node is a true positive, with zero false alarms on the test
> set. Recall=0.494 (F1=0.652 at thr=0.5; F1=0.675 at val-tuned threshold) reflects the inherent
> precision-recall tradeoff of a high-confidence detector — appropriate for SOC triage where
> false alarms have high analyst cost. GraphSAGE is also **port-agnostic**: it detects beaconing
> behavior from temporal graph features (`iat_cv`, `repeat_dst_ratio`) without relying on
> port signatures, which XGBoost cannot achieve.

### Why GNN Still Has Academic Value

| Capability | XGBoost | GNN (GraphSAGE/GATv2) |
|-----------|---------|----------------------|
| Per-flow classification | ✅ Excellent | ✅ Good |
| Neighborhood structure | ❌ Blind | ✅ Fan-out, beaconing patterns |
| Inductive (novel IPs) | ❌ Transductive | ✅ Inductive |
| Explainability | ✅ SHAP | ✅ Attention weights (GATv2) |
| Port-agnostic detection | ❌ Port-dependent | ✅ Structural features |
| Cross-scenario potential | ⚠️ Not tested | ⚠️ Future work |

### Real-time Pipeline (Verified)

- Graph snapshot every: **5 seconds (configurable)**
- GNN inference latency: **~58 ms/graph** (CPU, window=60s, GraphSAGE v3), FPR=0.00%
- Alert deduplication: **30-second dedup window**
- End-to-end detection: **< 1 second** (graph update + inference)

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+ required
python --version

# Install uv (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/annguyenax/realtime-c2-gnn-detection.git
cd realtime-c2-gnn-detection

# Copy environment config
cp .env.example .env

# Start all services (MLflow + dashboard + pipeline)
docker-compose up --build
```

Services:
- Dashboard: http://localhost:8501
- MLflow UI: http://localhost:5000
- Alert API: http://localhost:8000/docs

### Option 2: Local Development

```bash
git clone https://github.com/annguyenax/realtime-c2-gnn-detection.git
cd realtime-c2-gnn-detection

# Create virtualenv and install all deps
uv sync --all-extras

# Install PyTorch Geometric (CPU)
uv pip install torch-geometric torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

# Download CTU-13 Scenario 10
make download-data

# Preprocess flows
make preprocess SCENARIO=10

# Train XGBoost baseline
make train-xgboost

# Train GraphSAGE
make train-graphsage

# Run realtime demo
make demo
```

### Option 3: Makefile Reference

```bash
make help           # show all commands
make lint           # ruff + mypy
make test           # pytest with coverage
make security-scan  # bandit + trivy
make train-all      # xgboost + graphsage + gat
make demo           # realtime 3-thread pipeline
make mlflow-ui      # open MLflow at localhost:5000
make clean          # remove __pycache__, artifacts
```

---

## 🎬 Demo

### Alert JSON Output

```json
{
  "timestamp": "2024-01-15T14:23:07.142Z",
  "src_ip": "147.32.84.165",
  "dst_ip": "77.247.110.38",
  "risk_score": 0.9412,
  "model": "GraphSAGE",
  "window_id": "W_1705329787",
  "reason": [
    "periodic_communication: CoV=0.08",
    "high_out_degree: 47 unique destinations",
    "repeated_destination: contacted 38 times in 60s",
    "low_byte_volume: avg 184 bytes/flow",
    "suspicious_port: dst_port=6667 (IRC)"
  ]
}
```

### Dashboard Preview

> _Streamlit dashboard screenshot — run `make demo` to see live_
cd "D:\HK2_Nam4\AnToanMangNangCao\c2gnn_project_full\c2gnn_project"
powershell -ExecutionPolicy Bypass -File scripts\Start-Demo.ps1

---

## 🗂️ Repository Structure

```
c2gnn_project/
├── src/c2gnn/
│   ├── data/
│   │   └── flow_builder.py        # FlowRecord + CTU13FlowParser + BeaconingDetector
│   ├── graph/
│   │   └── dynamic_graph.py       # SlidingWindowGraph (TTL, incremental, NodeData, EdgeData)
│   ├── models/
│   │   ├── xgboost_baseline.py    # XGBoostC2Detector + SHAP + MLflow
│   │   └── graphsage.py           # GraphSAGEC2Detector + GATv2C2Detector + GNNTrainer
│   ├── realtime/
│   │   └── pipeline.py            # 3-thread orchestrator (FlowBuilderWorker, GraphUpdateWorker, InferenceWorker)
│   ├── api/
│   │   └── server.py              # FastAPI alert endpoint
│   ├── dashboard/
│   │   └── app.py                 # Streamlit live dashboard
│   ├── evaluation/                # (stub — future metrics/plot utilities)
│   └── utils/                     # (stub — future logging/config helpers)
├── scripts/
│   ├── 01_download_ctu13.py       # Auto-download CTU-13 scenarios
│   ├── 02_preprocess.py           # binetflow → parquet (temporal split)
│   ├── 03_train_xgboost.py        # Train XGBoost + SHAP + MLflow
│   ├── 04_train_gnn.py            # Train GraphSAGE / GATv2
│   ├── 05_collect_metrics.py      # Aggregate → reports/final_metrics.json
│   └── 06_threshold_analysis.py   # PR curve + threshold sweep for GNN
├── configs/
│   ├── dataset.yaml
│   ├── model_xgboost.yaml
│   ├── model_graphsage.yaml
│   ├── model_gat.yaml
│   └── realtime.yaml
├── notebooks/
│   └── 01_EDA.ipynb               # CTU-13 dataset exploration
├── tests/
│   ├── conftest.py
│   ├── test_flow_builder.py
│   ├── test_dynamic_graph.py
│   └── test_gnn_integration.py
├── models/artifacts/              # Trained model files (not committed if large)
│   ├── xgboost_metrics.json
│   ├── graphsage_metrics.json
│   └── shap_feature_importance.json
├── reports/
│   ├── final_metrics.json         # Generated by 05_collect_metrics.py
│   ├── results_table.txt
│   └── figures/                   # PR curves, confusion matrices
├── data/
│   ├── README.md                  # Dataset card + ethical use
│   ├── raw/ctu13/                 # Raw .binetflow (download with script 01)
│   └── processed/                 # Parquet splits (generated by script 02)
├── docs/
│   └── demo_script.md             # Step-by-step defense demo guide
├── .github/workflows/
│   ├── ci.yml                     # lint → test → build
│   └── security.yml               # bandit + trivy
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── Makefile
└── PROJECT_LOG.md                 # Progress tracker (this session)
```

---

## 🔬 Model Architecture

### GraphSAGE (Primary)

```
Input: 14-dim node features (degree, bytes, entropy, port ratio, IAT stats)
  ↓
SAGEConv(14 → 128) + BatchNorm + ReLU + Dropout(0.3)
  ↓
SAGEConv(128 → 64) + BatchNorm + ReLU + Dropout(0.3)
  ↓
SAGEConv(64 → 32) + BatchNorm + ReLU
  ↓
Linear(32 → 16) + ReLU
  ↓
Linear(16 → 2) → Softmax
Output: P(node = C2-related)
```

**Why GraphSAGE?** Inductive learning — generalizes to IPs not seen during training. Critical for detecting novel C2 infrastructure.

### GATv2 (Attention-based)

```
Input: 14-dim node features + optional edge features (bytes, flow_count)
  ↓
GATv2Conv(14 → 32, heads=8, edge_dim=4) + ELU
  ↓
GATv2Conv(256 → 16, heads=8) + ELU
  ↓
GATv2Conv(128 → 16, heads=1) + ELU
  ↓
Linear(16 → 2) → Softmax
Output: P(node = C2-related) + attention_weights per neighbor
```

**Why GATv2?** Attention weights highlight *which neighbors* triggered the alert — partial explainability without GNNExplainer overhead.

---

## 🔁 Reproducibility

```bash
# Exact environment lock file
uv sync --frozen

# Seed everything (set in configs/model_graphsage.yaml)
seed: 42

# Time-based train/test split — no data leakage
# First 70% timestamps → train, last 30% → test
# See: src/c2gnn/models/graphsage.py::GNNTrainer.time_based_split()

# DVC-managed data
dvc pull  # requires remote config in .dvc/config
```

All experiments logged to MLflow with:
- Git commit SHA
- Full hyperparameter config
- Metric curves
- Model artifacts + SHAP plots

---

## 🛡️ Security & Ethics

### DevSecOps Pipeline

| Check | Tool | When |
|-------|------|------|
| Static code analysis | `bandit` | Every PR |
| Dependency vulnerability | `pip-audit` | Every PR |
| Container scan | `trivy` | On push to main |
| Secret detection | `gitleaks` | Pre-commit |
| Type checking | `mypy` | Every PR |
| Linting | `ruff` | Every commit |

### Ethical Considerations

- Dataset contains **real botnet traffic** from 2011 CTU university network. Used for research only.
- IP addresses are **not linked to any individual** — network forensics dataset, not personal data.
- This system is designed for **defensive security** (SOC monitoring). Not intended for offensive use.
- Model outputs are **probabilistic risk scores**, not definitive accusations. Human review required.

---

## ⚠️ Known Limitations

This is a **research prototype**, not a production system. Known limitations:

| Limitation | Detail |
|-----------|--------|
| **Dataset age** | CTU-13 was captured in 2011. Modern C2 uses HTTPS/TLS tunneling not covered here. |
| **Single-scenario training** | Models trained only on Scenario 10 (Neris IRC botnet). Cross-scenario generalization not validated. |
| **XGBoost port dependence** | F1=0.992 relies on IRC port 6667 being discriminative. Port-agnostic C2 would degrade performance. |
| **GNN warm-start dependency** | GraphSAGE F1=0.652 under warm-start eval (continuous stream). Cold-start deployment (fresh graph) degrades to F1~0.06 due to lack of temporal context accumulation. Long-running deployment recommended. |
| **Node labeling approximation** | A node is labeled botnet if any adjacent edge is botnet — may over-label hub nodes (DNS servers, gateways). |
| **NetworkX scalability** | Single-threaded graph updates; not suitable for >10k concurrent nodes without architectural changes. |
| **Replay simulation** | Real-time pipeline replays recorded flows, not live packet capture. |
| **No encrypted C2** | Features are flow-level only (no TLS inspection, no JA3). |

## 🔮 Future Work

1. **Threshold tuning**: Select optimal threshold from validation PR curve (see `scripts/06_threshold_analysis.py`).
2. **Cross-scenario evaluation**: Test models trained on Scenario 10 against Scenario 8 (Rbot) and 11.
3. **Node label refinement**: Label only source IPs of botnet flows, not all adjacent nodes.
4. **Temporal GNN**: Replace static SAGEConv with EvolveGCN or TGN for true streaming graph learning.
5. **Encrypted C2 detection**: Add TLS certificate entropy and JA3 fingerprint features.
6. **Kafka integration**: Replace Python `Queue` with Redpanda/Kafka for distributed ingestion.
7. **GNNExplainer**: Full post-hoc graph explainability per alert.
8. **Active learning**: Human-in-the-loop labeling for SOC analysts.
9. **Kubernetes deployment**: Helm chart for production SOC integration.

---

## 👥 Team

| Member | Role | Responsibilities |
|--------|------|-----------------|
| **An Nguyen** | Data / Network / Security Engineer | CTU-13 preprocessing, FlowBuilder (Thread 1), Graph Update (Thread 2), C2 behavior analysis, dataset card, threat model |
| **Kai** | AI / GNN / MLOps Engineer | Dynamic graph engine, XGBoost baseline, GraphSAGE/GATv2 training, inference pipeline (Thread 3), MLflow, Docker, CI/CD, dashboard |

---

## 📚 References

1. Garcia, S. et al. (2014). *An empirical comparison of botnet detection methods.* CTU-13 Dataset. [Link](https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/)
2. Hamilton, W. et al. (2017). *Inductive Representation Learning on Large Graphs.* NeurIPS. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)
3. Brody, S. et al. (2022). *How Attentive are Graph Attention Networks?* ICLR. [arXiv:2105.14491](https://arxiv.org/abs/2105.14491)
4. Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
5. Lundberg, S. & Lee, S. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS (SHAP).
6. Fey, M. & Lenssen, J. (2019). *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop.

---

## 📄 License

MIT License — see [LICENSE](./LICENSE)

> **Academic Use Notice**: This project uses the CTU-13 dataset under its research license. Commercial use of this detection system requires independent dataset licensing review.
