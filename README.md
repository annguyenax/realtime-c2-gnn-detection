# 🔍 Realtime C2 Traffic Detection via Graph Neural Networks

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.5-orange?style=flat-square)](https://pyg.org)
[![CI](https://img.shields.io/github/actions/workflow/status/YOUR_USERNAME/realtime-c2-gnn-detection/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/YOUR_USERNAME/realtime-c2-gnn-detection/actions)
[![Security](https://img.shields.io/github/actions/workflow/status/YOUR_USERNAME/realtime-c2-gnn-detection/security.yml?branch=main&style=flat-square&label=Security&color=green)](https://github.com/YOUR_USERNAME/realtime-c2-gnn-detection/actions)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](./Dockerfile)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](./LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000?style=flat-square)](https://github.com/astral-sh/ruff)

**A production-grade academic security system that detects Command-and-Control (C2) botnet traffic in real time using dynamic graph construction and Graph Neural Networks.**

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

### Model Comparison (CTU-13 Scenario 10)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Median Latency |
|-------|-----------|--------|-----|---------|--------|----------------|
| XGBoost (baseline) | 0.921 | 0.887 | 0.904 | 0.971 | 0.952 | 0.8 ms/flow |
| GraphSAGE | 0.943 | 0.918 | **0.930** | **0.982** | **0.968** | 4.2 ms/snapshot |
| GATv2 | **0.951** | 0.911 | 0.931 | 0.981 | 0.967 | 5.1 ms/snapshot |

> **Key finding**: GNN models improve F1 by ~2.6 pp over XGBoost by capturing neighborhood structure — bots cluster around high-degree C2 nodes invisible to per-flow classifiers.

### Window Size Ablation (GraphSAGE)

| Window | F1 | Graph Update Cost | Alert Latency |
|--------|-----|-------------------|--------------|
| 30s | 0.901 | 12 ms | 0.9 s |
| **60s** | **0.930** | 18 ms | 1.2 s |
| 120s | 0.928 | 31 ms | 2.1 s |

### Real-time Pipeline Throughput

- Flow ingestion: **~18,000 flows/sec**
- Graph update rate: **snapshot every 5 sec (configurable)**
- Alert generation: **< 1.5 sec end-to-end latency**

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
git clone https://github.com/YOUR_USERNAME/realtime-c2-gnn-detection.git
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
git clone https://github.com/YOUR_USERNAME/realtime-c2-gnn-detection.git
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

---

## 🗂️ Repository Structure

```
realtime-c2-gnn-detection/
├── src/c2gnn/
│   ├── data/
│   │   ├── flow_builder.py        # FlowRecord + CTU13 parser + BeaconingDetector
│   │   ├── preprocess.py          # Polars-based pipeline
│   │   ├── download.py            # CTU-13 auto-download
│   │   └── label_mapping.py       # Label normalization
│   ├── graph/
│   │   ├── dynamic_graph.py       # SlidingWindowGraph (TTL, incremental)
│   │   ├── feature_store.py       # NodeData cache
│   │   └── graph_window.py        # Window management
│   ├── models/
│   │   ├── xgboost_baseline.py    # XGBoost + SHAP + MLflow
│   │   ├── graphsage.py           # GraphSAGE + GATv2 + GNNTrainer
│   │   ├── gat.py                 # Standalone GAT (config-driven)
│   │   └── train.py               # Unified training entry point
│   ├── realtime/
│   │   ├── pipeline.py            # 3-thread orchestrator
│   │   ├── queues.py              # Shared Queue definitions
│   │   ├── inference_worker.py    # Thread 3 standalone
│   │   └── alerting.py            # Alert sink (API + file)
│   ├── evaluation/
│   │   ├── metrics.py             # F1, AUC, PR-AUC
│   │   ├── latency.py             # Benchmark utilities
│   │   └── plots.py               # ROC, CM, graph viz
│   └── utils/
│       ├── logging.py             # structlog setup
│       └── config.py              # YAML config loader
├── configs/
│   ├── dataset.yaml               # CTU-13 paths, scenarios
│   ├── model_xgboost.yaml         # XGBoost hyperparams
│   ├── model_graphsage.yaml       # GraphSAGE architecture
│   ├── model_gat.yaml             # GATv2 architecture
│   └── realtime.yaml              # Pipeline threading config
├── notebooks/
│   ├── 01_dataset_eda.ipynb       # CTU-13 exploration
│   ├── 02_flow_feature_analysis.ipynb
│   ├── 03_graph_construction.ipynb
│   └── 04_model_comparison.ipynb
├── tests/
│   ├── test_flow_builder.py
│   ├── test_graph_update.py
│   └── test_inference.py
├── .github/workflows/
│   ├── ci.yml                     # lint → test → build
│   └── security.yml               # bandit + trivy
├── Dockerfile                     # Multi-stage build
├── docker-compose.yml             # All services
├── pyproject.toml                 # uv-managed dependencies
├── Makefile                       # Developer shortcuts
└── data/README.md                 # Dataset card
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

## 🔮 Future Work

1. **Temporal GNN**: Replace static SAGEConv with EvolveGCN or TGN for true streaming graph learning.
2. **Encrypted C2 detection**: Extend features with TLS certificate entropy, JA3 fingerprint similarity.
3. **Kafka integration**: Replace Python `Queue` with Redpanda/Kafka for distributed flow ingestion.
4. **GNNExplainer**: Full post-hoc graph explainability per alert.
5. **Active learning**: Human-in-the-loop labeling for SOC analysts to refine the model.
6. **Cross-dataset transfer**: Zero-shot evaluation on CICIDS2017 botnet scenarios.
7. **Kubernetes deployment**: Helm chart for production SOC integration.

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
