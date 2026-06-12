# Slide Outline — 14 Slides
# Realtime C2 Traffic Detection using Dynamic Graph Learning

---

## Slide 1: Title

**Title:** Realtime C2 Traffic Detection using Dynamic Graph Learning

**Subtitle:** Phát hiện Command-and-Control Traffic bằng Dynamic Graph Neural Networks

**Info:** Author | University | Date | Supervisor

---

## Slide 2: Vấn đề — C2 Traffic là gì?

**Nội dung chính:**
- Botnet C&C (Command-and-Control): máy nhiễm mã độc kết nối định kỳ đến C2 server để nhận lệnh
- Đặc điểm đặc trưng: **beaconing** — kết nối lặp lại, inter-arrival time đều đặn
- Gap hiện tại: signature-based IDS (port/IP blacklist) bị bypass khi C2 đổi port/domain

**Key question:** Làm thế nào phát hiện C2 mà không cần biết trước port hay IP?

**Hình:** Mô hình botnet: Bot → C2 Server → Attacker; timeline beaconing đều đặn

---

## Slide 3: Threat Model

**Nội dung:**
- **In scope:** IRC-based botnet beaconing, port-independent behavioral detection
- **Out of scope:** Encrypted C2 (HTTPS/TLS), DGA domain rotation, CDN fronting
- **Attacker capability:** Có thể đổi port nhưng không thể hoàn toàn che giấu periodicity nếu dùng automated beaconing
- **Defender goal:** Alert khi IP có beaconing pattern bất thường, với FPR < 0.1%

---

## Slide 4: Dataset — CTU-13 Scenario 10

**Nội dung:**
- Dataset: CTU-13 (Czech Technical University, 2011) — real botnet traffic, Murlo IRC C2
- Format: Bidirectional Argus flows (`.binetflow`) — NOT standard NetFlow v5/v9
- Citation: Garcia et al., *Computers & Security*, 2014

| Property | Value |
|---|---|
| Total flows | 5,178,417 |
| Botnet flows | 322,158 (6.22%) |
| Imbalance ratio | ~15:1 (flow-level), ~200:1 (node-level) |

- **Limitation rõ:** Dataset 2011, Murlo IRC — không đại diện C2 hiện đại (HTTPS, DGA)

**Hình:** Pie chart: 93.78% normal vs 6.22% botnet

---

## Slide 5: Kiến trúc hệ thống

**Nội dung:** 3-thread realtime pipeline

```
Flow Source → FlowBuilderWorker → GraphUpdateWorker → GNNInferenceWorker → Alert API
                                       ↕
                               Dynamic Graph (NetworkX)
                                       ↕
                               Streamlit Dashboard
```

**Các module:**
1. Flow Parser — đọc `.binetflow`, parse timestamp/IP/port/label
2. Sliding Window Graph Builder — window=60s, overlap=30s
3. GraphSAGE Inference — node classification, threshold=0.9118
4. FastAPI Alert API — POST `/alerts`, GET `/stats`
5. Streamlit Dashboard — realtime metrics + graph viz

---

## Slide 6: Dynamic Graph Construction

**Nội dung:**
- Mỗi **60 giây** = 1 graph snapshot
- **Node** = IP address (unique trong window)
- **Edge** = ít nhất 1 flow giữa 2 IPs trong window
- **Node features (18-dim):**
  - Flow stats (14): in/out bytes, in/out packets, flow count, protocols, flags...
  - **Temporal (4):** `active_span`, `mean_iat`, `iat_cv`, `repeat_dst_ratio`
- **Node label:** botnet=1 nếu bất kỳ adjacent edge là botnet flow

**Key:** `iat_cv` (coefficient of variation): beaconing → iat_cv ≈ 0.1; normal → iat_cv > 1.0

---

## Slide 7: Models — XGBoost và GraphSAGE

**XGBoost (baseline):**
- Input: 14 flow-level features (tabular), mỗi flow = 1 sample
- Task: Binary classification per flow
- Advantage: Fast, interpretable (SHAP), no graph needed

**GraphSAGE v3 (proposed):**
- Input: Dynamic graph snapshot, 18-dim node features
- Task: Node classification (IP-level botnet detection)
- Architecture: 3 SAGE layers, hidden=128, dropout=0.3
- Training: WeightedCE (cap=50), CosineAnnealingLR (T_max=100), patience=12

**SHAP Top Features (XGBoost):** `dst_port` (2.3), `src_port` (2.1), `bytes` (0.8)...

---

## Slide 8: Kết quả — XGBoost

| Metric | Value |
|---|---|
| F1 | **0.9921** |
| Precision | 0.9895 |
| Recall | 0.9947 |
| AUC | 0.9998 |
| FPR | 0.10% |
| Latency | ~2ms/flow |

**Giải thích tại sao XGBoost tốt:**
- CTU-13 Murlo botnet dùng IRC **port 6667** — feature cực kỳ discriminative
- SHAP: `dst_port` là feature #1
- Đây là signature-based detection ẩn trong tabular ML
- **Limitation:** Nếu botnet đổi port → XGBoost bị bypass

---

## Slide 9: Kết quả — GraphSAGE (Quan trọng: 2 threshold)

| | Default (thr=0.5) | Tuned (thr=0.9118) |
|---|---|---|
| F1 | 0.3951 | **0.6328** |
| Precision | 0.2675 | **0.7106** |
| Recall | 0.7557 | 0.5703 |
| FPR | 0.107% | **0.012%** |
| AUC | **0.9817** | — |

**Honest explanation:**
- Default F1=0.395 KHÔNG có nghĩa model kém — do class imbalance node-level (~200:1)
- AUC=0.982 ↔ model ranking rất tốt
- Threshold=0.9118 được tìm trên **validation set** (không phải test) → không leakage
- **FPR giảm 9× từ default sang tuned threshold**

---

## Slide 10: Threshold Analysis

**Hình:** PR Curve + Threshold Sweep

- PR-AUC = 0.6485 (tốt với 200:1 imbalance)
- Optimal threshold = 0.9118 (tối đa F1 trên validation với constraint recall ≥ 0.40)
- Với threshold = 0.9118: Precision 71%, Recall 57%, FPR 0.012%

**Key insight:** Trong bài toán bảo mật, FPR thấp ưu tiên hơn recall cao — thay vì 107 false alarms/1M nodes, chỉ còn 12 false alarms/1M nodes.

---

## Slide 11: Real-time Pipeline Demo

**Nội dung:**
- 3-thread architecture: build graph → inference → alert (concurrent)
- Replay dataset với `--realtime-factor 50` (50× faster than real time)
- Inference latency: **56ms/graph snapshot** (well within 30s window interval)
- Detection latency: 30–60 giây (window accumulation + inference)

**Demo command:**
```powershell
python -m c2gnn.realtime.pipeline `
  --data data/processed/scenario10_test.parquet `
  --model models/artifacts/graphsage_best.pt `
  --threshold 0.9118 --realtime-factor 50
```

---

## Slide 12: Dashboard Demo

**Screenshot/Live:** Streamlit dashboard tại `http://localhost:8501`

**Hiển thị:**
- Total alerts, FPR%, latency (realtime update)
- Alert table: timestamp, suspicious IP, confidence score
- Model comparison: XGBoost vs GraphSAGE (default + tuned)
- Graph visualization: node coloring theo botnet probability
- Timeline: alerts theo thời gian

---

## Slide 13: Limitations + Future Work

**Limitations:**
1. CTU-13 2011 — không đại diện HTTPS/TLS C2, DGA, CDN fronting
2. Single-scenario training — chưa validate cross-scenario (Sc.10 → Sc.8)
3. Cold-start gap: warm-start F1=0.652 vs cold-start F1=0.06 (graph cần warm-up)
4. XGBoost cao do port-dependent signature — không phải generalizable detection
5. NetworkX backend — không scale > 10k concurrent nodes

**Future Work:**
1. Cross-scenario evaluation (train Sc.10, test Sc.8 Rbot)
2. Encrypted flow features (TLS metadata, byte entropy)
3. Warm graph initialization để giải quyết cold-start gap
4. Distributed graph store (GraphBolt, DGL distributed) cho production scale
5. Post-processing k/n smoothing để tăng precision thêm

---

## Slide 14: Kết luận

**Đóng góp chính:**
1. Complete end-to-end pipeline: từ raw CTU-13 flows đến realtime GNN alert
2. GraphSAGE với 18-dim temporal features: F1_tuned=0.633, FPR=0.012%
3. Temporal beaconing features (iat_cv, repeat_dst_ratio) cải thiện F1 từ 0.399 → 0.633
4. Honest analysis: cold-start gap, port-dependent XGBoost limitation, threshold transparency
5. Research prototype — không claim production SOC-ready

**Take-away:**
- Graph-based detection bổ sung cho tabular ML: topology-aware, port-agnostic, inductive
- AUC=0.982 chứng minh graph learning detect được botnet behavioral pattern
- Threshold tuning là bước bắt buộc cho imbalanced node classification

**Định vị:** Research prototype + Real-time demo | Không phải SOC-ready product
