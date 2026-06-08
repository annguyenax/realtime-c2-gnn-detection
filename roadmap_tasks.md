# 🗺️ Roadmap 6 Tuần — realtime-c2-gnn-detection

> **Nguyên tắc:** Sau mỗi sprint (tuần), phải có deliverable cụ thể,
> commit lên GitHub, checklist xanh hết. Không để nợ kỹ thuật sang tuần sau.

---

## 👥 Phân công cố định

| Role | Thành viên | GitHub Label |
|------|-----------|--------------|
| **Member 1 — Data/Network/Security Engineer** | An | `m1-data` |
| **Member 2 — AI/GNN/MLOps Engineer** | Kai | `m2-ml` |

> Cả hai cùng làm: architecture design, báo cáo, slide, demo, review code.

---

## 📅 WEEK 1 — Project Setup + Dataset Understanding
> **Mục tiêu:** Repo xịn, môi trường chạy được, hiểu rõ CTU-13.

### Cả hai (Day 1–2)
- [ ] Tạo GitHub repo `realtime-c2-gnn-detection` (public)
- [ ] Init project bằng `uv init`, setup `pyproject.toml`
- [ ] Setup branch: `main`, `develop`, tất cả `feature/*` branch từ `develop`
- [ ] Setup pre-commit hooks (ruff, mypy, bandit)
- [ ] Viết `Makefile` với target: `setup`, `test`, `lint`, `train`, `demo`
- [ ] Tạo `.env.example`, `.gitignore`, `LICENSE`
- [ ] Thiết kế kiến trúc hệ thống (vẽ diagram, lưu vào `docs/architecture.md`)
- [ ] Tạo GitHub Project Board với Kanban columns

### Member 1 — Data (Day 2–7)
- [ ] Tải CTU-13 Scenario 1, 8, 10 (3 kịch bản botnet khác nhau: IRC, P2P, HTTP)
  - Download link: https://www.stratosphereips.org/datasets-ctu13
  - Ưu tiên: Scenario 10 (Neris) cho training chính
- [ ] Đọc kỹ README CTU-13: hiểu format `.binetflow`, label conventions
- [ ] Phân tích label distribution (Normal vs Botnet vs Background)
- [ ] Notebook `01_dataset_eda.ipynb`: thống kê cơ bản, visualize label imbalance
- [ ] Viết `CTU13FlowParser` (đọc `.binetflow` → `FlowRecord`)
- [ ] Unit test parser: `tests/test_flow_builder.py`

### Member 2 — AI/MLOps (Day 2–7)
- [ ] Setup MLflow tracking server (local, `mlflow ui`)
- [ ] Setup DVC với local remote: `dvc init`, `dvc remote add`
- [ ] Viết `pyproject.toml` đầy đủ với tất cả dependencies
- [ ] Setup GitHub Actions CI: lint + test (`.github/workflows/ci.yml`)
- [ ] Setup GitHub Actions Security: bandit + trivy (`.github/workflows/security.yml`)
- [ ] Đọc tài liệu PyTorch Geometric: hiểu `Data`, `SAGEConv`, `GATv2Conv`
- [ ] Viết `configs/model_graphsage.yaml` và `configs/dataset.yaml`

**Deliverable cuối tuần 1:**
- ✅ Repo public, CI xanh, pre-commit pass
- ✅ Notebook EDA với visualize dataset
- ✅ `FlowRecord` dataclass và parser viết xong

---

## 📅 WEEK 2 — Data Pipeline + XGBoost Baseline
> **Mục tiêu:** Data pipeline hoàn chỉnh, XGBoost chạy được, có metrics đầu tiên.

### Member 1 — Data (Day 8–14)
- [ ] Hoàn thiện `flow_builder.py`:
  - Normalize tất cả label (botnet/normal/background)
  - Handle missing values, port hex conversion
  - Lọc background flows có thể exclude hoặc giữ tùy chiến lược
- [ ] Viết `preprocess.py`:
  - Feature engineering: `bytes_per_packet`, `fwd_bwd_ratio`, port entropy
  - Xử lý class imbalance: log scale_pos_weight
  - Export: `data/processed/ctu13_scenario10_flows.parquet`
- [ ] Viết `label_mapping.py`: mapping label về binary (botnet=1, else=0)
- [ ] Notebook `02_flow_feature_analysis.ipynb`:
  - Phân tích feature importance (correlation heatmap)
  - Phân tích C2 behavior: beaconing, low volume, repeated dst
  - Visualize flow duration, byte rate distribution botnet vs normal

### Member 2 — AI/MLOps (Day 8–14)
- [ ] Viết `models/xgboost_baseline.py` (code skeleton đã có)
- [ ] Train XGBoost với 80/20 split, StratifiedKFold=5
- [ ] Log experiment lên MLflow: params + metrics + model artifact
- [ ] Generate SHAP summary plot, save vào `reports/figures/`
- [ ] Viết `evaluation/metrics.py`: F1, AUC, PR-AUC, confusion matrix
- [ ] Viết `evaluation/plots.py`: ROC curve, PR curve, confusion matrix plot
- [ ] Ghi kết quả vào `reports/tables/baseline_results.md`

**Deliverable cuối tuần 2:**
- ✅ Data pipeline end-to-end: binetflow → parquet
- ✅ XGBoost F1 > 0.85 (expected ~0.93 với CTU-13)
- ✅ SHAP plot, ROC curve đẹp
- ✅ MLflow experiment logged

---

## 📅 WEEK 3 — Dynamic Graph Construction
> **Mục tiêu:** Graph builder hoàn chỉnh, visualize graph, chuẩn bị data cho GNN.

### Member 1 — Data (Day 15–21)
- [ ] Viết `graph/dynamic_graph.py`:
  - `SlidingWindowGraph` với TTL-based edge expiry
  - Incremental update (không rebuild toàn bộ)
  - `NodeData.to_feature_vector()` đủ 14 features
  - `EdgeData.update()` với flow aggregation
- [ ] Viết `graph/feature_store.py`:
  - Cache node features để tránh recompute
  - Support lookup theo IP
- [ ] Viết `graph/graph_window.py`:
  - Quản lý nhiều window size (30s, 60s, 120s)
  - Convert `SlidingWindowGraph` → `PyG Data`

### Member 2 — AI/MLOps (Day 15–21)
- [ ] Notebook `03_graph_construction.ipynb`:
  - Visualize graph snapshot tại các thời điểm khác nhau (NetworkX + PyVis)
  - Phân tích graph metrics: degree distribution, clustering coefficient
  - Compare window size 30s vs 60s vs 120s
  - Visualize botnet IPs trong graph (red nodes)
- [ ] Viết `evaluation/latency.py`:
  - Benchmark graph construction time per window
  - Benchmark incremental update vs full rebuild
  - Plot latency vs number of nodes/edges
- [ ] Unit test: `tests/test_graph_update.py`

**Deliverable cuối tuần 3:**
- ✅ Dynamic graph hoạt động với TTL + incremental update
- ✅ Graph visualization đẹp (botnet IPs highlight)
- ✅ Benchmark: incremental update < 50ms với 1000 nodes
- ✅ `PyG Data` object export thành công

---

## 📅 WEEK 4 — GNN Training + Comparison
> **Mục tiêu:** GraphSAGE + GAT train xong, so sánh với XGBoost, có bảng kết quả.

### Member 1 — Data (Day 22–28)
- [ ] Viết script tạo full graph dataset từ CTU-13:
  - Split theo time: train windows (first 70%) / val (15%) / test (15%)
  - Lưu PyG `DataLoader` hoặc list of Data objects
  - Xử lý class imbalance cho node classification
  - Thêm scenario 8 làm generalization test
- [ ] Phân tích C2 behavior trong graph:
  - Beaconing: visualize inter-arrival time histogram
  - High out-degree botnet nodes
  - Repeated destination patterns
  - Viết `reports/c2_behavior_analysis.md`

### Member 2 — AI/MLOps (Day 22–28)
- [ ] Viết `models/graphsage.py` + `models/gat.py` (code skeleton đã có)
- [ ] Viết `models/train.py`:
  - Training loop với early stopping
  - Class-weighted CrossEntropyLoss
  - MLflow logging: loss, F1, AUC per epoch
  - Checkpoint saving
- [ ] Train GraphSAGE: 100 epochs, tune hidden_channels, dropout
- [ ] Train GATv2: 100 epochs, tune heads
- [ ] Notebook `04_model_comparison.ipynb`:
  - Bảng so sánh: XGBoost vs GraphSAGE vs GAT
  - ROC curve 3 models trên cùng figure
  - Visualize GAT attention weights (top-5 most attended edges)

**Deliverable cuối tuần 4:**
- ✅ GraphSAGE F1 > XGBoost (expected improvement ~3-8%)
- ✅ Bảng so sánh đầy đủ: F1, AUC, PR-AUC, inference time
- ✅ GAT attention visualization
- ✅ MLflow: 3 experiments với đầy đủ metrics

---

## 📅 WEEK 5 — Realtime Pipeline + Dashboard
> **Mục tiêu:** 3-thread pipeline chạy demo được, dashboard hiển thị alert.

### Member 1 — Data (Day 29–35)
- [ ] Viết `realtime/pipeline.py` — Thread 1 (FlowBuilderWorker)
- [ ] Viết `realtime/queues.py` — định nghĩa queue với metrics
- [ ] Viết `realtime/alerting.py`:
  - Alert JSON format chuẩn
  - Alert to file, alert to FastAPI endpoint
  - Dedup logic (không alert cùng IP 2 lần trong 30s)
- [ ] Viết FastAPI app: `src/c2gnn/api.py`
  - GET `/alerts` — list recent alerts
  - GET `/graph/snapshot` — current graph stats
  - POST `/config/threshold` — update detection threshold

### Member 2 — AI/MLOps (Day 29–35)
- [ ] Viết Thread 2 (`GraphUpdateWorker`) + Thread 3 (`InferenceWorker`)
- [ ] Tích hợp TorchScript export cho inference nhanh hơn
- [ ] Viết Streamlit dashboard `src/c2gnn/dashboard.py`:
  - Real-time alert feed
  - Graph statistics (nodes/edges over time)
  - Detection timeline
  - Risk score distribution
- [ ] Benchmark latency end-to-end:
  - Flow arrival → alert: measure P50/P95/P99 latency
  - Target: P95 < 500ms
- [ ] Viết `tests/test_inference.py`

**Deliverable cuối tuần 5:**
- ✅ Demo chạy được: phát lại CTU-13, alert hiển thị realtime
- ✅ Dashboard Streamlit đẹp
- ✅ Latency benchmark report
- ✅ API endpoint hoạt động

---

## 📅 WEEK 6 — Hardening + Report + Slide + Demo
> **Mục tiêu:** Project hoàn thiện, report nộp, slide đẹp, demo recording.

### Cả hai (Day 36–42)
- [ ] **Docker:**
  - Viết `Dockerfile` multi-stage (builder + runtime)
  - Viết `docker-compose.yml` (app + mlflow + streamlit)
  - Test: `docker compose up` chạy full pipeline
- [ ] **DevSecOps hardening:**
  - Trivy scan Docker image (0 CRITICAL)
  - Semgrep scan codebase
  - Bandit: no HIGH severity
  - Add security scan vào CI
- [ ] **README:**
  - Badges: Python, CI, Docker, License, MLflow
  - Architecture diagram (Mermaid)
  - Demo GIF (record bằng `asciinema` hoặc screen recorder)
  - Results table (copy từ notebook)
- [ ] **Report (~30-40 trang):** viết theo outline trong `reports/final_report.md`
- [ ] **Slide (15-20 trang):** theo `slides/presentation_outline.md`
- [ ] **Model card:** `models/MODEL_CARD.md`
- [ ] **Dataset card:** `data/DATASET_CARD.md`
- [ ] **Threat model:** `docs/THREAT_MODEL.md`
- [ ] Record demo video (3-5 phút)
- [ ] Code review lẫn nhau, fix linting, coverage > 70%

**Deliverable cuối tuần 6:**
- ✅ GitHub repo public, star-worthy
- ✅ Report nộp đúng format
- ✅ Slide đẹp, demo GIF trong README
- ✅ CI xanh hết, Docker build thành công
- ✅ README.md professional (EN)

---

## 📊 Milestone Checklist Summary

| Milestone | Week | Owner | Status |
|-----------|------|-------|--------|
| Repo init + CI setup | 1 | Both | ⬜ |
| CTU-13 parsed + EDA | 1-2 | M1 | ⬜ |
| XGBoost baseline | 2 | M2 | ⬜ |
| Dynamic graph | 3 | M1 | ⬜ |
| Graph dataset export | 3-4 | M1 | ⬜ |
| GraphSAGE + GAT trained | 4 | M2 | ⬜ |
| Model comparison table | 4 | Both | ⬜ |
| Realtime pipeline 3 threads | 5 | Both | ⬜ |
| Dashboard + API | 5 | Both | ⬜ |
| Docker + security scan | 6 | M2 | ⬜ |
| Report + Slide | 6 | Both | ⬜ |
| Demo recording | 6 | Both | ⬜ |

---

## ⚠️ Risk & Mitigation

| Risk | Probability | Mitigation |
|------|------------|------------|
| CTU-13 download chậm | Medium | Download sớm từ tuần 1, dùng gdrive mirror |
| PyG compatibility với CUDA | Medium | Test trên CPU trước, note GPU optional |
| GNN không outperform XGBoost | Low | Expected improvement small but justifiable by inductive learning |
| Graph quá lớn, OOM | Low | Limit to 500 nodes per window for demo |
| Không có GPU | Medium | Train trên Google Colab, export model |
