# PROMPT TẠO BÁO CÁO ĐỒ ÁN — PHÁT HIỆN C2 TRAFFIC BẰNG GRAPH LEARNING

> Copy toàn bộ phần dưới đây vào Claude / ChatGPT để tạo báo cáo hoàn chỉnh.

---

```
Bạn là chuyên gia viết báo cáo học thuật chuyên ngành An toàn thông tin.
Hãy viết đầy đủ một báo cáo đồ án môn học theo chuẩn học thuật Việt Nam,
dựa hoàn toàn vào thông tin kỹ thuật thực tế được cung cấp bên dưới.
KHÔNG được bịa đặt số liệu, thêm thí nghiệm không có trong dữ liệu.

═══════════════════════════════════════════════════════════════════════
PHẦN 0: THÔNG TIN ĐỊNH DẠNG BÁO CÁO
(Áp dụng định dạng chuẩn đồ án môn học PTIT TP.HCM)
═══════════════════════════════════════════════════════════════════════

Font chữ toàn bộ: Times New Roman
Cỡ chữ nội dung chính: 13pt
Cỡ chữ tiêu đề chương: 14pt, in đậm, IN HOA
Cỡ chữ tiêu đề mục (1.1, 1.2...): 13pt, in đậm
Cỡ chữ tiêu đề mục con (1.1.1...): 13pt, in đậm nghiêng
Cỡ chữ chú thích hình/bảng: 11pt, canh giữa
Cỡ chữ trong bảng: 12pt

Lề trang:
- Lề trái: 3.5 cm
- Lề phải: 2 cm
- Lề trên: 2.5 cm
- Lề dưới: 2.5 cm

Giãn dòng: 1.5 lines
Thụt đầu đoạn: 1.27 cm (Tab đầu tiên)
Khoảng cách giữa các đoạn (Spacing After): 6pt

Đánh số trang: số arabic, canh giữa, phía dưới, bắt đầu từ trang nội dung (sau trang bìa, mục lục)
Đánh số hình: Hình 1.1, Hình 2.1... (số chương.số thứ tự)
Đánh số bảng: Bảng 1.1, Bảng 2.1...
Đánh số công thức: (1.1), (1.2)...

Tiêu đề chương cách nội dung: Spacing Before 24pt, Spacing After 12pt
Mỗi chương bắt đầu trang mới (Page Break trước)

CÁCH TRÌNH BÀY BÁO CÁO THEO MẪU CHUẨN:
- Trang bìa: Logo trường, tên trường, tên môn, tên đề tài, SV, GVHD, năm
- Trang bìa phụ: giống bìa chính
- Mục lục tự động
- Danh mục hình vẽ
- Danh mục bảng biểu
- Danh mục từ viết tắt
- Nội dung báo cáo (Chương 1–7)
- Tài liệu tham khảo (IEEE style)
- Phụ lục (nếu có)

═══════════════════════════════════════════════════════════════════════
PHẦN 1: THÔNG TIN ĐỀ TÀI
═══════════════════════════════════════════════════════════════════════

Tên đề tài:
  "Phát hiện luồng Command-and-Control (C2) bằng Graph Learning đáp ứng thời gian thực"
  (Tiếng Anh: Realtime C2 Traffic Detection using Dynamic Graph Learning)

Môn học: An toàn mạng nâng cao
Trường: Học viện Công nghệ Bưu chính Viễn thông – Cơ sở tại TP.HCM (PTIT)
Học kỳ: HK2 năm học 2025–2026
Giảng viên hướng dẫn: [Tên GVHD]
Sinh viên thực hiện: [Tên sinh viên]

Mục tiêu đồ án (từ đề bài):
1. Phát hiện luồng liên lạc C2 của botnet dựa trên cấu trúc giao tiếp IP–IP và đặc trưng packet/flow.
2. Phân tích gói tin/flow để tạo edge (src IP → dst IP) theo thời gian.
3. Xây dựng graph động theo cửa sổ thời gian; node features = thống kê flow/packet.
4. Xử lý đa luồng: Thread 1 (Flow builder), Thread 2 (Graph update), Thread 3 (GNN inference + alert).
5. So sánh: ML baseline (XGBoost) vs GraphSAGE/GAT.
6. Đánh giá: F1/AUC, detection time, chi phí cập nhật graph.
7. Phân tích lợi ích của graph đối với C2; đề xuất cách cập nhật graph nhẹ để giảm latency.

GitHub: https://github.com/annguyenax/realtime-c2-gnn-detection

═══════════════════════════════════════════════════════════════════════
PHẦN 2: DỮ LIỆU KỸ THUẬT THỰC TẾ (dùng trực tiếp vào báo cáo)
═══════════════════════════════════════════════════════════════════════

── 2A. DATASET ──────────────────────────────────────────────────────

Tên dataset: CTU-13 Botnet Dataset
Nguồn: Czech Technical University (CTU), Prague – Stratosphere IPS Lab
Trang mô tả: https://www.stratosphereips.org/datasets-ctu13
Nguồn tải: https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/
Citation: Garcia, S., Grill, M., Stiborek, J., & Zunino, A. (2014).
  "An empirical comparison of botnet detection methods."
  Computers & Security, 45, 100–123. DOI: 10.1016/j.cose.2014.05.011
Năm thu thập: 2011
Định dạng: Bidirectional Argus flows (.binetflow) – KHÔNG phải NetFlow v5/v9

Kịch bản sử dụng: Scenario 10 – Murlo IRC C2 botnet

Thống kê dataset (đã xác minh):
  Tổng số flows:         5.178.417
  Flows botnet (C2):       322.158  (6,22%)
  Flows bình thường:     4.856.259  (93,78%)
  Tỷ lệ mất cân bằng (flow-level):  ~15:1
  Tỷ lệ mất cân bằng (node-level):  ~200:1 (sau khi xây graph)

Split theo thời gian (temporal – không shuffle):
  Train:      70% flows đầu tiên  (≈3.624.892 flows)
  Validation: 15% tiếp theo       (≈776.763 flows) – dùng để tune threshold
  Test:       15% cuối cùng       (≈776.762 flows) – đánh giá chính thức

── 2B. KIẾN TRÚC HỆ THỐNG ──────────────────────────────────────────

Pipeline 3 luồng song song (multi-threaded):

  [CTU-13 .binetflow / .parquet]
           │
           ▼
  ┌─────────────────────────────┐
  │  Thread 1: FlowBuilderWorker│  ← parse flows, tạo FlowRecord
  │  Tốc độ: replay × factor   │    theo thứ tự thời gian
  └────────────┬────────────────┘
               │ flow_queue (bounded)
               ▼
  ┌─────────────────────────────┐
  │  Thread 2: GraphUpdateWorker│  ← sliding window 60s
  │  Xây dựng dynamic graph     │    node/edge aggregation
  │  Snapshot mỗi 30s (50% OL) │    xuất graph snapshots
  └────────────┬────────────────┘
               │ graph_queue (bounded)
               ▼
  ┌──────────────────────────────┐
  │  Thread 3: InferenceWorker   │  ← GNN forward pass
  │  GraphSAGE / GATv2           │    node classification P(botnet)
  │  Alert nếu score > threshold │    gửi alert qua FastAPI
  └────────────┬─────────────────┘
               │ HTTP POST /alerts
               ▼
  ┌────────────────────────────────────────┐
  │  FastAPI Alert API  │  Streamlit Dashboard │
  │  POST /alerts       │  Realtime metrics    │
  │  GET  /stats        │  Alert table, graph  │
  │  GET  /alerts       │  viz, model comparison│
  └────────────────────────────────────────┘

Thư viện và framework:
  - Python 3.11+, PyTorch 2.3, PyTorch Geometric 2.5
  - XGBoost 2.x, scikit-learn, SHAP
  - FastAPI + Uvicorn, Streamlit
  - NetworkX (graph backend), pandas, numpy
  - MLflow (experiment tracking), Docker (deployment)
  - CI/CD: GitHub Actions (ruff lint + pytest + Bandit + Trivy)

── 2C. NODE FEATURES (18 chiều) ─────────────────────────────────────

Flow stats (14 features):
  1.  in_flows          – Số flows đến node trong window
  2.  out_flows         – Số flows đi từ node trong window
  3.  in_bytes          – Tổng bytes nhận
  4.  out_bytes         – Tổng bytes gửi
  5.  in_packets        – Tổng packets nhận
  6.  out_packets       – Tổng packets gửi
  7.  unique_srcs       – Số IP nguồn duy nhất
  8.  unique_dsts       – Số IP đích duy nhất
  9.  tcp_ratio         – Tỷ lệ TCP
  10. udp_ratio         – Tỷ lệ UDP
  11. mean_duration     – Thời lượng flow trung bình (giây)
  12. std_duration      – Độ lệch chuẩn thời lượng
  13. suspicious_port   – Tỷ lệ cổng đáng ngờ (IRC 6667, >49152)
  14. fan_out_ratio     – out_flows / (in_flows + out_flows)

Temporal beaconing features (4 features – mới trong v3):
  15. active_span       – Khoảng thời gian node hoạt động trong window (s)
  16. mean_iat          – IAT trung bình giữa các flows (s)
  17. iat_cv            – Coefficient of variation của IAT
                          (C2 beaconing: iat_cv ≈ 0.05–0.25; bình thường: >1.0)
  18. repeat_dst_ratio  – Tỷ lệ flows gửi đến đích đã gửi trước đó

── 2D. MODEL ARCHITECTURE ───────────────────────────────────────────

XGBoost:
  Input:       14 flow-level tabular features (không dùng graph)
  Task:        Binary classification per flow
  n_estimators: 400, max_depth: 7, scale_pos_weight: ~15
  CV:          5-fold Stratified CV (chỉ để ước lượng variance)
  Evaluation:  Temporal split (không shuffle)

GraphSAGE v3:
  Input:       18-dim node features + graph topology
  Task:        Node classification (mỗi IP = một node)
  Architecture:
    Layer 1: SAGEConv(18 → 128) + BatchNorm + ReLU + Dropout(0.3)
    Layer 2: SAGEConv(128 → 128) + BatchNorm + ReLU + Dropout(0.3)
    Layer 3: SAGEConv(128 → 64) + BatchNorm + ReLU
    Output:  Linear(64 → 1) + Sigmoid
  Loss:        WeightedCrossEntropy (max_class_weight=50)
  Optimizer:   Adam(lr=0.001)
  Scheduler:   CosineAnnealingLR(T_max=100, eta_min=1e-5)
  Training:    epochs=50, patience=12 (early stopping)
  Seed:        42 (reproducible)
  Inductive:   Có – xử lý được IP mới không thấy trong training

GATv2:
  Input:       18-dim node features
  Architecture: 2 × GATv2Conv(heads=4, hidden=64)
  Chú thích:   Untuned – dùng làm baseline so sánh attention mechanism

── 2E. KẾT QUẢ THỰC NGHIỆM (SỐ LIỆU CHÍNH THỨC) ──────────────────

⚠️ QUAN TRỌNG: Phải trình bày CẢ HAI threshold cho GNN:
   - Threshold mặc định (0.5): minh bạch về hiện tượng class imbalance
   - Threshold tuned từ validation: kết quả thực tế khi triển khai

┌─────────────────────┬───────────┬──────────┬──────────┬─────────┬──────────┬────────────┐
│ Model               │ Threshold │ F1       │ Precision│ Recall  │ AUC-ROC  │ FPR        │
├─────────────────────┼───────────┼──────────┼──────────┼─────────┼──────────┼────────────┤
│ XGBoost             │ 0.5       │ 0.9921   │ 0.9895   │ 0.9947  │ 0.9998   │ 0.10%      │
│ GraphSAGE           │ 0.5       │ 0.3951   │ 0.2675   │ 0.7557  │ 0.9817   │ 0.107%     │
│ GraphSAGE (tuned)   │ 0.9118    │ 0.6328   │ 0.7106   │ 0.5703  │ 0.9817   │ 0.012%     │
│ GATv2               │ 0.5       │ 0.0518   │ 0.0267   │ 0.8389  │ 0.9701   │ 1.537%     │
└─────────────────────┴───────────┴──────────┴──────────┴─────────┴──────────┴────────────┘

PR-AUC (GraphSAGE): 0.6485 (tốt với imbalance 200:1 ở node-level)
Latency:
  XGBoost:   2.1 ms/flow
  GraphSAGE: 56.2 ms/graph snapshot
  GATv2:     296.5 ms/graph snapshot

SHAP Top 5 (XGBoost):
  src_port:         2.811  ← feature phân biệt nhất
  bytes_per_packet: 1.997
  dst_port:         1.997
  is_tcp:           1.416
  total_bytes:      1.254

Threshold tuning (GraphSAGE):
  Phương pháp: tối đa hóa F1 trên validation set, constraint min_recall=0.40
  Optimal threshold: 0.9118
  Lý do threshold cao: class imbalance nặng → model calibration bị shift
  KHÔNG phải information leakage: threshold tìm trên val, evaluate trên test

Cold-start gap (finding quan trọng):
  Warm-start (continuous stream): F1 = 0.652
  Cold-start (fresh empty graph): F1 = 0.06
  Root cause: temporal features (iat_cv, repeat_dst_ratio) cần lịch sử tích lũy

── 2F. DEMO EVIDENCE ────────────────────────────────────────────────

Pipeline đã chạy thành công:
  - 3.000 flows → 7 graph snapshots
  - 7 GNN inferences hoàn chỉnh
  - Inference latency: 18.38 ms (sample alert)
  - 53 real alerts trong reports/alerts.jsonl

Alert mẫu thực tế:
{
  "timestamp_iso": "2011-08-18T07:13:14Z",
  "src_ip": "147.32.84.25",
  "dst_ip": "195.113.232.96",
  "risk_score": 0.3063,
  "model": "GraphSAGEC2Detector",
  "reasons": ["short-lived connections (possible beaconing)"],
  "inference_latency_ms": 18.38,
  "graph_nodes": 437,
  "graph_edges": 710
}

FastAPI endpoints: /health, /stats, /alerts (đã chạy, HTTP 200 confirmed)
Streamlit dashboard: http://localhost:8501 (đã chạy)

── 2G. HÌNH ẢNH SẴN CÓ (chèn vào báo cáo) ─────────────────────────

reports/figures/dataset_distribution.png    – Pie + bar chart phân bố nhãn
reports/figures/model_comparison_bar.png    – So sánh F1/Prec/Rec/AUC 4 model
reports/figures/shap_importance.png         – SHAP top 10 features (XGBoost)
reports/figures/fpr_comparison.png          – So sánh FPR các model
reports/figures/node_features_table.png     – Bảng 18 node features
reports/figures/latency_comparison.png      – So sánh latency (ms)
reports/figures/pr_curve_graphsage.png      – PR curve GraphSAGE
reports/figures/threshold_sweep_graphsage.png – Threshold sweep (F1/Prec/Rec)

═══════════════════════════════════════════════════════════════════════
PHẦN 3: CẤU TRÚC BÁO CÁO (7 CHƯƠNG)
═══════════════════════════════════════════════════════════════════════

Viết đầy đủ, chi tiết từng chương. Mỗi chương tối thiểu 3–5 trang.
Tổng báo cáo mục tiêu: 50–70 trang (không kể phụ lục).

── CHƯƠNG 1: GIỚI THIỆU (4–5 trang) ────────────────────────────────

1.1 Đặt vấn đề
  - C2 (Command-and-Control) botnet: định nghĩa, vai trò trong tấn công APT
  - Tại sao C2 nguy hiểm: kiểm soát tập trung hàng triệu bot, exfiltration, ransomware
  - Đặc điểm beaconing khiến C2 khó phát hiện: low-volume, periodic, looks normal
  - Hạn chế IDS truyền thống: signature-based bị bypass khi C2 đổi port/domain

1.2 Động lực nghiên cứu
  - Per-flow ML (XGBoost) không thấy topology: 1 flow isolated looks benign
  - Graph-based approach: IP relationships encode botnet pattern tốt hơn
  - Bài toán cụ thể: phát hiện bot node trong dynamic IP graph

1.3 Mục tiêu và phạm vi
  - Mục tiêu: xây dựng hệ thống phát hiện C2 realtime dùng GNN
  - Phạm vi: CTU-13 Scenario 10 (Murlo IRC), proof-of-concept
  - Không phải: production SOC system, phát hiện C2 HTTPS hiện đại

1.4 Đóng góp của đề tài
  1. Pipeline realtime 3 luồng: FlowBuilder → GraphUpdate → GNNInference
  2. Temporal beaconing features (iat_cv, repeat_dst_ratio) cải thiện F1 từ 0.399 → 0.633
  3. So sánh đầy đủ: XGBoost vs GraphSAGE vs GATv2 trên cùng dataset và evaluation protocol
  4. Phân tích cold-start gap – finding quan trọng cho chiến lược deployment
  5. Honest reporting: phân biệt rõ default threshold và tuned threshold

1.5 Cấu trúc báo cáo
  - Mô tả ngắn nội dung từng chương (7 chương)

── CHƯƠNG 2: CƠ SỞ LÝ THUYẾT (6–8 trang) ──────────────────────────

2.1 Mạng botnet và giao tiếp C2
  - Mô hình botnet: bot, bot herder, C2 server
  - Giao thức C2: IRC (port 6667), HTTP, HTTPS, DNS, P2P
  - Beaconing: định nghĩa, đặc trưng IAT, CoV
  - Ví dụ: Murlo (IRC), Neris (IRC), Rbot (HTTP)

2.2 Graph Neural Networks
  - Giới thiệu GNN, message passing framework
  - GraphSAGE (Hamilton et al., 2017): inductive, NeighborSampling aggregation
  - GATv2 (Brody et al., 2022): dynamic attention cải tiến từ GAT
  - Tại sao GNN phù hợp C2: inductive learning, topology-aware, handles unseen nodes

2.3 Class imbalance và threshold tuning
  - Imbalance: 15:1 flow-level, ~200:1 node-level trong graph
  - Xử lý: class weights, filter empty snapshots
  - ROC-AUC vs F1: ROC-AUC đo ranking, F1 phụ thuộc threshold
  - Threshold tuning: tối đa hóa F1 trên validation set, không phải test set

2.4 Sliding window graph
  - Định nghĩa: window T giây, snapshot interval T/2 (50% overlap)
  - Node = IP address, edge = ≥1 flow giữa 2 IP trong window
  - Temporal features: cần tích lũy ≥2 flows để tính iat_cv

2.5 Các công trình liên quan
  - [cite] Euler (Hamilton et al., 2020): Temporal GNN cho fraud detection
  - [cite] E-GraphSAGE (Lo et al., 2022): GNN cho IDS trên CTU-13
  - [cite] Xu et al., 2022: Dynamic graph cho botnet detection
  - So sánh phương pháp này với related work

── CHƯƠNG 3: DATASET VÀ TIỀN XỬ LÝ DỮ LIỆU (5–6 trang) ───────────

3.1 Giới thiệu CTU-13
  - Czech Technical University, 2011
  - 13 kịch bản, botnet gia đình khác nhau
  - Format: bidirectional Argus flows (.binetflow) – khác NetFlow v5/v9
  - Citation đầy đủ: Garcia et al., 2014

3.2 Scenario 10 – Murlo IRC C2
  - Mô tả botnet Murlo: IRC protocol, port 6667, beaconing interval ~30-45s
  - Thống kê: 5.178.417 flows, 6.22% botnet
  - Cấu trúc file .binetflow: timestamp, duration, protocol, src/dst IP, port, flags, bytes, packets, label

3.3 Tiền xử lý
  - Parse .binetflow → pandas DataFrame
  - Label mapping: "Botnet" → 1, "Background/Normal/LEGITIMATE" → 0
  - Xử lý missing values và outliers
  - Temporal split: 70% train / 15% val / 15% test (không shuffle)
  - Lý do temporal split: tránh data leakage, phản ánh deployment thực tế

3.4 Xây dựng graph
  - Sliding window: 60 giây
  - Node labeling: node = botnet nếu BẤT KỲ edge nào là botnet flow
  - Known limitation: có thể over-label hub nodes (DNS, gateway)
  - filter_empty=True: loại bỏ snapshot không có botnet node trong training
  - Kết quả: 421 snapshots (train: 295, val: 63, test: 63)

3.5 Bảng thống kê dataset (chèn Hình dataset_distribution.png)

── CHƯƠNG 4: FEATURE ENGINEERING VÀ KIẾN TRÚC MODEL (7–9 trang) ───

4.1 Node feature vector (18 chiều)
  - Bảng đầy đủ 18 features (chèn Hình node_features_table.png)
  - Phân tích ý nghĩa từng nhóm
  - Temporal features: tại sao iat_cv quan trọng cho C2 detection

4.2 XGBoost – Baseline flow-level classifier
  - Input: 14 flow-level features
  - Architecture và hyperparameters
  - Scale_pos_weight để xử lý imbalance
  - SHAP analysis (chèn Hình shap_importance.png)
  - Giải thích tại sao XGBoost cao: src_port và dst_port discriminative với IRC 6667

4.3 GraphSAGE v3
  - Kiến trúc chi tiết (3 SAGEConv layers)
  - Inductive learning: xử lý unseen nodes trong production
  - WeightedCrossEntropy với max_class_weight=50
  - CosineAnnealingLR: T_max=100, lý do chọn T_max lớn hơn epochs

4.4 GATv2
  - Kiến trúc (2 layers, 4 heads)
  - Attention mechanism: tự học neighbor quan trọng
  - Hạn chế: chưa tune threshold, là baseline so sánh

4.5 Threshold tuning
  - Vấn đề F1 mặc định thấp: class imbalance node-level ~200:1
  - Phương pháp: sweep threshold [0.01, 0.99] trên validation set
  - Constraint: min_recall=0.40 để không bỏ sót quá nhiều botnet
  - Kết quả: optimal_threshold=0.9118 (chèn Hình threshold_sweep_graphsage.png)
  - Tính hợp lệ: threshold tìm trên val, evaluate chính thức trên test

── CHƯƠNG 5: HỆ THỐNG PHÁT HIỆN THỜI GIAN THỰC (5–6 trang) ─────────

5.1 Kiến trúc tổng quan
  - Vẽ/mô tả pipeline 3-thread (dùng sơ đồ từ mục 2B)
  - Luồng dữ liệu: flow → graph snapshot → alert
  - Bounded queue: tránh memory overflow

5.2 Thread 1 – FlowBuilderWorker
  - Parse .parquet / .binetflow theo thứ tự thời gian
  - Tham số --realtime-factor để điều chỉnh tốc độ replay
  - Output: FlowRecord objects → flow_queue

5.3 Thread 2 – GraphUpdateWorker
  - Sliding window 60 giây, snapshot interval 30 giây
  - Incremental graph update: O(1) per flow (không rebuild)
  - Tính toán 18 node features sau mỗi snapshot
  - Output: PyTorch Geometric Data objects → graph_queue

5.4 Thread 3 – InferenceWorker
  - Load model từ checkpoint (.pt file)
  - Forward pass: 56.2 ms/snapshot (GraphSAGE, CPU)
  - Alert generation: P(botnet) > threshold → POST /alerts
  - Alert deduplication: tránh alert trùng lặp cùng IP

5.5 FastAPI Alert API
  - Endpoints: POST /alerts, GET /stats, GET /alerts, GET /health
  - Alert JSON schema
  - Ví dụ alert thực tế (từ mục 2F)

5.6 Streamlit Dashboard
  - Realtime metrics: total_alerts, FPR estimate, latency
  - Alert table, timeline, model comparison panel
  - Screenshot dashboard (chèn screenshot nếu có)

── CHƯƠNG 6: THỰC NGHIỆM VÀ ĐÁNH GIÁ (8–10 trang) ─────────────────

6.1 Thiết lập thực nghiệm
  - Hardware: CPU (không dùng GPU), Windows 11 Pro
  - Dataset: CTU-13 Scenario 10, temporal split 70/15/15
  - Metrics: F1, Precision, Recall, AUC-ROC, PR-AUC, FPR, Latency

6.2 Kết quả XGBoost
  - Bảng kết quả đầy đủ
  - SHAP analysis: tại sao src_port quan trọng nhất
  - Nhận xét: F1=0.992 cao nhờ IRC port 6667 discriminative
  - QUAN TRỌNG: đây là signature-based detection ẩn trong tabular ML

6.3 Kết quả GraphSAGE – phân tích dual-threshold
  - BẢNG 2 DÒNG:
    Default threshold (0.5):  F1=0.395, AUC=0.982, FPR=0.107%
    Tuned threshold (0.9118): F1=0.633, Prec=0.711, Rec=0.570, FPR=0.012%
  - Giải thích hiện tượng: AUC cao + F1 default thấp = class imbalance calibration
  - PR curve analysis (chèn Hình pr_curve_graphsage.png)
  - Temporal features đóng góp: F1 tăng từ 0.399 (v2) → 0.633 (v3)

6.4 Kết quả GATv2
  - F1=0.052, AUC=0.970, FPR=1.537%
  - Nhận xét: FPR cao nhất – nhiều false alarms nhất
  - Cần tune threshold để cải thiện (future work)

6.5 So sánh tổng hợp
  - Bảng tổng hợp đầy đủ (chèn Hình model_comparison_bar.png)
  - FPR comparison (chèn Hình fpr_comparison.png)
  - Latency comparison (chèn Hình latency_comparison.png)

6.6 Phân tích: Vì sao XGBoost tốt hơn GNN trên CTU-13?
  - CTU-13 Murlo dùng IRC port 6667 → XGBoost nhận ra signature
  - GNN không "học" port trực tiếp, học topology pattern
  - Kết luận: XGBoost là port-dependent, GNN là behavior-dependent
  - Nếu C2 đổi port → XGBoost sẽ bị degraded, GNN ổn định hơn

6.7 Cold-start gap analysis
  - Warm-start F1 = 0.652 vs Cold-start F1 = 0.06
  - Nguyên nhân: temporal features cần tích lũy graph history
  - Hàm ý deployment: cần warm-up period 60-120 giây trước khi alert

6.8 Đánh giá realtime pipeline
  - Throughput: 3.000 flows trong 2.2 giây (1.363 flows/s)
  - 7 graph snapshots, 7 inferences
  - Detection latency: ~18-56 ms/snapshot
  - End-to-end: 30-60 giây (window accumulation + inference)

── CHƯƠNG 7: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (3–4 trang) ─────────────

7.1 Kết luận
  - Tóm tắt đóng góp (4 điểm từ mục 1.4)
  - Đánh giá theo tiêu chí đồ án:
    ✅ Multi-threading: FlowBuilder / GraphUpdate / GNNInference hoạt động song song
    ✅ Dynamic graph: sliding window 60s, incremental update
    ✅ So sánh XGBoost vs GraphSAGE vs GATv2
    ✅ Đánh giá F1/AUC/FPR/Latency đầy đủ
    ✅ Phân tích lợi ích graph: topology-aware, port-agnostic
    ✅ Realtime pipeline với latency 56ms
  - Định vị sản phẩm: research prototype + realtime demo, KHÔNG phải production SOC

7.2 Hạn chế
  - CTU-13 năm 2011: không đại diện C2 hiện đại (HTTPS, DGA, CDN fronting)
  - Single-scenario training: chưa validate cross-scenario generalization
  - Cold-start gap: F1 giảm từ 0.652 → 0.06 với fresh graph
  - XGBoost port-dependent: F1 cao không generalizable
  - NetworkX: không scale lên >10k concurrent nodes

7.3 Hướng phát triển
  - Cross-scenario evaluation: train Scenario 10 → test Scenario 8 (Rbot)
  - Temporal GNN (TGN): học time-aware graph evolution
  - Warm graph initialization: giảm cold-start gap
  - Kafka/Flink: distributed streaming cho production scale
  - Post-processing: k/n alert smoothing để tăng precision thêm
  - Dataset mới hơn: CICIDS2017, UNSW-NB15, CIC-IDS2018

═══════════════════════════════════════════════════════════════════════
PHẦN 4: TÀI LIỆU THAM KHẢO (IEEE format, ≥15 references)
═══════════════════════════════════════════════════════════════════════

Bắt buộc có (dùng IEEE numbering [1], [2]...):

[1] S. Garcia, M. Grill, J. Stiborek, and A. Zunino, "An empirical comparison
    of botnet detection methods," Computers & Security, vol. 45, pp. 100-123, 2014.
    DOI: 10.1016/j.cose.2014.05.011

[2] W. Hamilton, Z. Ying, and J. Leskovec, "Inductive Representation Learning
    on Large Graphs," in Advances in Neural Information Processing Systems
    (NeurIPS), 2017, pp. 1024–1034.

[3] S. Brody, U. Alon, and E. Yahav, "How Attentive are Graph Attention
    Networks?" in International Conference on Learning Representations (ICLR), 2022.

[4] A. Lo, W. C. Loyola-González, W. G. Morales, and J. R. Fontes, "E-GraphSAGE:
    A Graph Neural Network based Intrusion Detection System for IoT,"
    in IEEE/IFIP Network Operations and Management Symposium (NOMS), 2022.

[5] T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph
    Convolutional Networks," in ICLR, 2017.

[6] P. Velickovic et al., "Graph Attention Networks," in ICLR, 2018.

[7] Y. Zhou, G. Cheng, and J. Yu, "Graph Neural Networks: A Review of Methods
    and Applications," AI Open, vol. 1, pp. 57–81, 2020.

[8] H. Yao et al., "Botnet Detection Based on P2P Traffic Behavior Analysis,"
    IEEE Access, vol. 7, pp. 184799–184814, 2019.

[9] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System,"
    in ACM SIGKDD, 2016, pp. 785–794.

[10] S. M. Lundberg and S. I. Lee, "A Unified Approach to Interpreting Model
     Predictions," in NeurIPS, 2017, pp. 4765–4774.

[11] H. Staudemeyer and E. Morris, "Understanding LSTM – a tutorial into Long
     Short-Term Memory Recurrent Neural Networks," arXiv:1909.09586, 2019.

[12] Stratosphere IPS Lab, "CTU-13 Dataset." [Online]. Available:
     https://www.stratosphereips.org/datasets-ctu13. [Accessed: Jun. 2026].

[13] MCFP Lab, "CTU Malware Capture Facility Project." [Online]. Available:
     https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/. [Accessed: Jun. 2026].

[14] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward Generating
     a New Intrusion Detection Dataset and Intrusion Traffic Characterization,"
     in ICISSP, 2018, pp. 108–116. (CICIDS2017)

[15] M. Ring et al., "A Survey of Network-based Intrusion Detection Data Sets,"
     Computers & Security, vol. 86, pp. 147–167, 2019.

═══════════════════════════════════════════════════════════════════════
PHẦN 5: YÊU CẦU VIẾT
═══════════════════════════════════════════════════════════════════════

1. Ngôn ngữ: tiếng Việt học thuật, chính xác, không dùng từ ngữ thông thường.
   Ví dụ: "thực hiện" thay vì "làm", "được đề xuất" thay vì "nói ra".

2. Viết ĐÚNG số liệu đã cung cấp. KHÔNG sửa, KHÔNG làm tròn khác đi.
   Ví dụ: F1=0.6328 KHÔNG viết thành "F1≈0.63" hay "F1=0.65".

3. Mỗi kết quả quan trọng phải có câu giải thích nguyên nhân.
   Ví dụ: "GraphSAGE đạt F1=0.3951 với threshold mặc định do hiện tượng
   mất cân bằng lớp nghiêm trọng ở mức node (~200:1), khiến model
   phân loại với xác suất thấp hơn 0.5 cho hầu hết các node botnet..."

4. Phân biệt rõ kết quả default threshold và tuned threshold.
   - Không được chỉ report F1=0.395 mà không giải thích.
   - Không được chỉ report F1=0.633 mà không nói rõ threshold là 0.9118.

5. Thừa nhận hạn chế trực tiếp, không che giấu:
   - XGBoost cao vì port-dependent, không generalizable.
   - CTU-13 cũ, không đại diện C2 hiện đại.
   - GNN cần warm-up, cold-start F1 chỉ 0.06.

6. Đặt tài liệu tham khảo [cite] theo IEEE tại mỗi vị trí cần thiết.

7. Viết đầy đủ tất cả 7 chương theo cấu trúc đã mô tả.
   Không viết tóm tắt ngắn, viết ĐẦY ĐỦ mỗi mục.

8. Chú thích hình: "Hình X.Y: [Mô tả hình], nguồn: tác giả"
   Chú thích bảng: "Bảng X.Y: [Mô tả bảng]"
```
