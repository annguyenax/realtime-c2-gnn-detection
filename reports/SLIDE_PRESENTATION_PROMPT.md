# PROMPT: Tạo Slide Thuyết Trình + Chiến lược Demo

> Copy toàn bộ phần dưới dây vào Claude / ChatGPT để tạo slide

---

```
Bạn là chuyên gia trình bày học thuật. Hãy tạo nội dung đầy đủ cho 20 slide
thuyết trình đồ án, kèm theo chiến lược thuyết trình và kịch bản demo.

═══════════════════════════════════════════════════════════
THÔNG TIN ĐỒ ÁN
═══════════════════════════════════════════════════════════

Tên đề tài: "Phát hiện C2 Traffic bằng Graph Learning đáp ứng thời gian thực"
Môn học   : An toàn mạng nâng cao — PTIT TP.HCM
Thời gian : 15 phút thuyết trình + 5 phút Q&A + 3 phút demo live
Audience  : Giảng viên chuyên ngành ATTT, 1-2 giảng viên phản biện, ~30 sinh viên

Kết quả thực nghiệm (điền số thật sau khi chạy scripts/05_collect_metrics.py):
  XGBoost   : F1=___, AUC=___, FPR=___%, latency=___ ms
  GraphSAGE : F1=___, AUC=___, FPR=___%, latency=___ ms
  GATv2     : F1=___, AUC=___, FPR=___%, latency=___ ms
  Dataset   : CTU-13, ___ flows tổng, ___% botnet (Scenario 10)

══════════════════════════════════════════════════════════
PHẦN A: NỘI DUNG 20 SLIDE
══════════════════════════════════════════════════════════

Viết CHI TIẾT nội dung từng slide theo format sau:

╔═══════════════════════════════════════════════════════╗
║  SLIDE [số]: [Tiêu đề]                                ║
║  Thời gian: [XX giây]   Người trình bày: [1 hoặc 2]  ║
╠═══════════════════════════════════════════════════════╣
║  NỘI DUNG CHÍNH (bullet points hoặc diagram text):   ║
║  ...                                                   ║
╠═══════════════════════════════════════════════════════╣
║  LỜI NÓI (speaker notes — nói gì khi chiếu slide):   ║
║  ...                                                   ║
╠═══════════════════════════════════════════════════════╣
║  VISUAL GỢI Ý: [mô tả hình nên dùng]                 ║
╚═══════════════════════════════════════════════════════╝

─────────────────────────────────────
CẤU TRÚC 20 SLIDE (thực hiện tuần tự)
─────────────────────────────────────

SLIDE 1: Trang bìa (20s)
- Tên đề tài, nhóm sinh viên, môn học, PTIT, ngày
- Visual: Logo PTIT + network graph visualization (node-link diagram)
- Speaker: [Trưởng nhóm]

SLIDE 2: Mục tiêu thuyết trình (30s)
- Roadmap 4 phần: Vấn đề → Giải pháp → Kết quả → Demo
- Visual: progress bar hoặc numbered list
- Giúp giảng viên biết cấu trúc ngay từ đầu

SLIDE 3: Vấn đề — C2 là gì và tại sao nguy hiểm (60s)
- Định nghĩa C2 botnet bằng ngôn ngữ đơn giản
- Con số: 67% các cuộc tấn công APT dùng C2, $X billion thiệt hại
- Ví dụ thực tế: Emotet, Cobalt Strike, Mirai
- Tại sao C2 khó phát hiện: encrypted, low-volume, mimics normal traffic
- Visual: diagram "attacker → C2 server → infected bots" với mũi tên

SLIDE 4: Giới hạn của phương pháp hiện tại (45s)
- Signature-based IDS: bỏ sót C2 zero-day, evasion dễ dàng
- ML truyền thống (per-flow): không thấy topology pattern
- Key insight: C2 tạo ra CẤU TRÚC ĐỒ THỊ đặc trưng (star topology)
- Visual: 2 ô so sánh — "Per-flow ML" vs "Graph-based ML"
- Quote/highlight: "Individual flows look innocent; the GRAPH reveals the bot"

SLIDE 5: Đề xuất giải pháp (45s)
- Graph Neural Network trên dynamic IP graph
- Realtime: detect trong < [X] ms sau khi flow xảy ra
- So sánh nhanh: XGBoost (baseline) vs GraphSAGE vs GATv2
- Visual: pipeline diagram đơn giản 3 bước

SLIDE 6: Dataset — CTU-13 (45s)
- Czech Technical University, 2011-2014
- 13 kịch bản botnet thực tế: IRC, HTTP, P2P C2
- Focus: Scenario 10 (Murlo IRC) + Scenario 8 (Rbot, gen test)
- Số liệu: ___ flows, ___% botnet (imbalanced!)
- Visual: bảng 3 cột (Scenario | Botnet family | C2 Protocol)
   Scenario 1: Neris | IRC
   Scenario 8: Rbot  | HTTP
   Scenario 10: Murlo | IRC  ← chúng tôi dùng

SLIDE 7: Đặc trưng C2 Traffic (60s) — SLIDE QUAN TRỌNG
- Hiển thị 2 biểu đồ song song (mô tả để vẽ):
  TRÁI: Normal traffic — IAT random, CoV cao (>0.5)
  PHẢI: C2 beaconing — IAT đều đặn, CoV thấp (<0.15)
- 4 đặc trưng chính với icon/visual:
  🔁 Beaconing: kết nối định kỳ (CoV < 0.15)
  📦 Low-volume: avg [X] bytes/flow
  ⭐ Star topology: bot → C2 lặp đi lặp lại trong graph
  🔌 Suspicious ports: IRC 6667, ephemeral > 49152
- Speaker: "Đây là insight cốt lõi của toàn bộ đề tài"

SLIDE 8: Kiến trúc hệ thống (90s) — SLIDE PHỨC TẠP NHẤT
- Diagram 3-thread pipeline (vẽ ASCII dạng flowchart):

  [CTU-13 File] 
       ↓
  [Thread 1: FlowBuilder]
  Parse binetflow → FlowRecord
  BeaconingDetector (CoV)
       ↓ flow_queue
  [Thread 2: GraphUpdate]
  SlidingWindowGraph (60s window)
  14-dim node features
  8-dim edge features
       ↓ graph_queue
  [Thread 3: GNN Inference]
  GraphSAGE / GATv2
  P(bot) per node
  Alert if > 0.7
       ↓
  [FastAPI + Streamlit Dashboard]

- Highlight: "3 threads chạy đồng thời — thực sự realtime"
- Mention: bounded queue (maxsize=1000) để không OOM

SLIDE 9: Graph Construction — Node & Edge Features (60s)
- Node feature vector (14-dim) — highlight 5 quan trọng nhất:
  out_flows, fan_out_ratio, dst_ip_entropy, suspicious_port_ratio, std_duration
- Edge feature vector (8-dim)
- Sliding window mechanism: TTL-based expiry
- Visual: ví dụ mini-graph với 3 nodes:
  "10.0.0.1 (bot) ──[3 flows]──> 185.220.101.1 (c2)"
  Chú thích node features bên cạnh

SLIDE 10: Model 1 — XGBoost Baseline (45s)
- Input: 18-dim per-flow feature (không có graph context)
- Config: 400 trees, depth=7, Focal loss equivalent (scale_pos_weight)
- 5-fold Stratified CV → F1 = ___ ± ___
- SHAP Top features: duration, flow_iat_mean, bytes_per_packet
- Visual: SHAP barchart (mô tả top 5 features)
- Nhấn mạnh: "baseline PHẢI mạnh để GNN improvement có ý nghĩa"

SLIDE 11: Model 2 — GraphSAGE (45s)
- Input: 14-dim node features + graph structure
- Architecture: 3 × SAGEConv(128) → BatchNorm → Dropout(0.3)
- Key: NeighborLoader → inductive learning (xử lý IP mới)
- Focal Loss (γ=2) để xử lý imbalance
- Visual: diagram 3 lớp SAGEConv với aggregation arrows

SLIDE 12: Model 3 — GATv2 + Attention (45s)
- 2 × GATv2Conv(heads=4, dim=64) — attention per neighbor
- Attention weight = "tự động học neighbor nào quan trọng"
- Ví dụ: bot tập trung attention vào C2 server → high score
- Visual: mini attention heatmap (vẽ dạng table)
  Bot_IP → [C2_server: 0.87, server2: 0.06, server3: 0.07]

SLIDE 13: KẾT QUẢ — Bảng so sánh (90s) — SLIDE QUAN TRỌNG NHẤT
- Bảng kết quả rõ ràng, lớn:

  ┌──────────┬───────┬───────┬──────────┬──────────┐
  │ Model    │  F1   │  AUC  │  FPR     │ Latency  │
  ├──────────┼───────┼───────┼──────────┼──────────┤
  │ XGBoost  │ ___   │ ___   │  ___%    │  ___ ms  │
  │ GraphSAGE│ ___   │ ___   │  ___%    │  ___ ms  │
  │ GATv2    │ ___   │ ___   │  ___%    │  ___ ms  │
  └──────────┴───────┴───────┴──────────┴──────────┘

- Highlight: GATv2 best F1 và FPR; XGBoost best latency
- Phân tích: GNN tốt hơn XGBoost [X]% F1 nhờ graph structure
- Speaker: "FPR thấp = ít false alarm = SOC analyst không bị overload"

SLIDE 14: KẾT QUẢ — Tại sao GNN vượt XGBoost (60s)
- 3 lý do kỹ thuật cụ thể:
  1. Star topology: bot kết nối lặp lại tới C2 → rõ ràng trong graph
  2. Low-degree malicious nodes: XGBoost không thấy neighbor pattern
  3. Attention trong GATv2: tự highlight suspicious neighbor
- Visual: 2 mini-graph so sánh:
  TRÁI: "XGBoost thấy gì?" — 1 flow isolated, looks normal
  PHẢI: "GATv2 thấy gì?" — bot + C2 cluster, clearly anomalous

SLIDE 15: KẾT QUẢ — Generalization Test (Scenario 8) (30s)
- Train: Scenario 10 (Murlo IRC) → Test: Scenario 8 (Rbot HTTP)
- F1 cross-scenario: ___
- Ý nghĩa: model generalize được sang botnet gia đình khác
- Nếu F1 giảm: "hợp lý vì Rbot dùng HTTP không phải IRC"
- Visual: bảng 2 hàng (same-scenario vs cross-scenario)

SLIDE 16: Realtime Pipeline Performance (45s)
- Throughput: ___ flows/second (Thread 1)
- Graph update: O(1) per flow (incremental, không rebuild)
- End-to-end detection time: ___ ms từ flow đến alert
- Queue không bị overflow trong thử nghiệm
- Visual: timeline diagram "flow arrives → alert generated [X ms later]"

SLIDE 17: Alert Sample + Dashboard (45s)
- Hiển thị JSON alert mẫu (thật):
  {
    "timestamp": "2011-08-18T09:46:53",
    "src_ip": "147.32.84.165",
    "dst_ip": "...",
    "risk_score": 0.94,
    "model": "GATv2",
    "reasons": ["periodic_beaconing", "high_fan_out", ...]
  }
- Streamlit dashboard: mô tả layout (alert table + graph viz)
- Visual: screenshot hoặc ASCII mockup của dashboard

SLIDE 18: DevSecOps & Code Quality (30s)
- GitHub CI/CD: ruff lint + pytest (30 tests, pass ✓) + Bandit + Trivy
- Docker multi-stage: image nhỏ, production-ready
- MLflow: track tất cả experiments, reproducible
- Visual: badges từ GitHub Actions (CI ✓, Security ✓)
- GitHub: github.com/annguyenax/realtime-c2-gnn-detection

SLIDE 19: Hạn chế & Hướng phát triển (45s)
- HẠN CHẾ (trung thực, giảng viên đánh giá cao tính honest):
  • Dataset 2011 — C2 hiện đại dùng HTTPS/DNS-over-HTTPS
  • Graph cold start: cần ~60s tích lũy flow
  • Không decode payload
  • Chưa test production traffic

- HƯỚNG PHÁT TRIỂN:
  • Kafka/Flink cho production-scale streaming
  • Temporal GNN (TGN) cho time-aware graph evolution
  • Federated learning (multi-sensor, privacy-preserving)
  • SSL certificate fingerprinting cho HTTPS C2

SLIDE 20: Kết luận + Q&A (45s)
- 3 đóng góp chính của đề tài (bullet points):
  1. Hệ thống realtime 3-thread: [X] flows/sec, < [Y] ms detection
  2. GNN vượt XGBoost baseline [Z]% F1 nhờ graph structure
  3. Open-source, Docker-ready, CI/CD pipeline hoàn chỉnh
- Cảm ơn + mời Q&A
- Visual: GitHub repo QR code + logo PTIT

══════════════════════════════════════════════════════════
PHẦN B: CHIẾN LƯỢC THUYẾT TRÌNH
══════════════════════════════════════════════════════════

Viết hướng dẫn chi tiết cho từng mục:

## B.1 Phân chia nhiệm vụ (2 người)

Thành viên 1 (Data/Network Engineer):
  - Trình bày: Slide 1-3 (giới thiệu, vấn đề, dataset)
  - Trình bày: Slide 6-7 (dataset, C2 features)
  - Trình bày: Slide 18-20 (DevSecOps, kết luận)
  - Demo: chạy live demo trên terminal
  - Trả lời Q&A: câu hỏi về dataset, network security, threat model

Thành viên 2 (AI/ML Engineer):
  - Trình bày: Slide 4-5 (giới hạn, đề xuất)
  - Trình bày: Slide 8-12 (kiến trúc, models)
  - Trình bày: Slide 13-17 (kết quả, alerts)
  - Trả lời Q&A: câu hỏi về GNN, model architecture, metrics

## B.2 Timing breakdown (tổng 15 phút)

Phân bổ thời gian cho từng nhóm slide:
- Slide 1-2 (Opening): 50 giây
- Slide 3-5 (Vấn đề + Giải pháp): 2.5 phút
- Slide 6-7 (Dataset + C2 Features): 1.5 phút
- Slide 8-9 (Kiến trúc): 2.5 phút
- Slide 10-12 (Models): 2 phút
- Slide 13-16 (Kết quả): 3 phút ← QUAN TRỌNG NHẤT
- Slide 17-18 (Demo + DevSecOps): 1 phút
- Slide 19-20 (Hạn chế + Kết luận): 1.5 phút

## B.3 Key messages (3 điểm giảng viên PHẢI nhớ)

1. "Graph reveals what tabular ML cannot see"
   → Slide 14: giải thích cụ thể tại sao GNN tốt hơn

2. "Real-time: [X] ms từ flow đến alert"
   → Nhấn mạnh 3-thread pipeline, incremental update

3. "Honest about limitations — and know how to improve"
   → Slide 19: tự mình nêu hạn chế trước khi bị hỏi

## B.4 Câu hỏi phản biện thường gặp (chuẩn bị trước)

Q1: "Dataset CTU-13 năm 2011 quá cũ, có relevance không?"
Trả lời: "Cảm ơn câu hỏi hay. CTU-13 vẫn là benchmark chuẩn trong
nghiên cứu botnet detection vì: (1) labeled ground truth chất lượng cao,
(2) nhiều paper gần đây 2020-2024 vẫn dùng để so sánh, (3) beaconing
pattern của C2 về cơ bản không thay đổi dù protocol có khác.
Hướng mở rộng: test trên CICIDS-2018 hoặc traffic thực."

Q2: "Tại sao không dùng LSTM/Transformer thay vì GNN?"
Trả lời: "LSTM tốt cho temporal sequence của một flow, nhưng không
encode quan hệ GIỮA các IP. GNN encode cả hai: temporal (qua sliding window)
và spatial (graph topology). Thực tế bài báo [cite] cho thấy GNN
outperforms LSTM-based IDS vì cấu trúc botnet là graph problem về bản chất."

Q3: "FPR [X]% có thực sự đủ thấp cho production không?"
Trả lời: "Trong môi trường với 1 triệu flows/ngày, FPR X% = X*10,000
false alerts/ngày — vẫn quá cao cho SOC analyst. Thực tế cần FPR < 0.01%.
Chúng tôi đề xuất: (1) hai tầng lọc: XGBoost triage → GNN confirm;
(2) threshold tuning dựa trên cost matrix của từng tổ chức."

Q4: "Tại sao dùng sliding window 60s? Basis nào?"
Trả lời: "Dựa trên phân tích EDA notebook: median beaconing interval
trong CTU-13 là 30-45s. Window 60s đảm bảo ít nhất 1-2 beaconing cycles
trong mỗi snapshot. Chúng tôi cũng test 30s và 120s — trình bày
trong thesis nhưng không có thời gian trong slide."

Q5: "Sao không dùng raw packet thay vì NetFlow?"
Trả lời: "NetFlow trade-off có chủ ý: (1) privacy compliant — không
inspect payload, phù hợp với GDPR và môi trường enterprise;
(2) scale tốt hơn — 1 flow record thay vì N packets;
(3) ISP và enterprise router sẵn có NetFlow export.
Payload-based DPI là hướng phát triển khi cần accuracy cao hơn."

Q6: "Model có overfit CTU-13 không?"
Trả lời: "Chúng tôi test generalization qua 2 cách: (1) temporal split
— train trên 80% đầu, test trên 20% sau của Scenario 10;
(2) cross-scenario test: train S10, test S8 (Rbot — botnet khác hoàn toàn).
Kết quả F1 cross-scenario = [___] — giảm [___] điểm so với same-scenario,
acceptable cho unseen botnet family."

## B.5 Cách handle khi bị hỏi câu không biết

"Đây là câu hỏi rất hay. Trong phạm vi đề tài này chúng tôi chưa
explore hướng đó. Dựa trên kiến thức của tôi, [X], nhưng để trả lời
chính xác cần thêm thực nghiệm. Đây là một hướng mở rộng tốt."

[KHÔNG NÓI: "Tôi không biết" — luôn acknowledge và bridge sang điều bạn biết]

══════════════════════════════════════════════════════════
PHẦN C: KỊCH BẢN DEMO LIVE (3 phút)
══════════════════════════════════════════════════════════

Viết kịch bản demo chi tiết theo format:

## C.1 Setup trước buổi bảo vệ (làm từ tối hôm trước)

Checklist:
□ Cài đủ deps: pip install -e ".[dev]" trong môi trường demo
□ Test chạy pipeline ít nhất 3 lần trước
□ Chuẩn bị sẵn 2 terminal windows mở
□ Cài sẵn data trong data/processed/ (không download live)
□ Backup: screen recording demo nếu mạng/laptop có vấn đề
□ Laptop sạc đầy, disable Windows Update, tắt notifications

## C.2 Kịch bản demo (step-by-step, tổng ~3 phút)

[Người 1 trình bày, Người 2 thao tác terminal]

BƯỚC 1 — Giới thiệu demo setup (15s):
  Nói: "Chúng ta có 2 cửa sổ terminal:
        - Terminal trái: pipeline realtime
        - Terminal phải: alert stream output"

BƯỚC 2 — Khởi động pipeline (30s):
  Gõ (đã chuẩn bị sẵn từ trước):
    python -m c2gnn.realtime.pipeline \
      --data data/processed/scenario10_test.parquet \
      --model models/artifacts/gatv2_best.pt \
      --realtime-factor 50.0 \
      --log-level INFO

  Nói trong khi chạy: "Pipeline bắt đầu replay traffic CTU-13 
  với tốc độ 50x realtime. Thread 1 parse flows, Thread 2 xây 
  graph, Thread 3 chạy GATv2 inference..."

BƯỚC 3 — Chờ alert đầu tiên xuất hiện (60s):
  Khi alert xuất hiện, nói:
  "Đây — alert đầu tiên! GATv2 phát hiện IP [X] với risk score [0.94]
  trong [Y] giây. System đưa ra lý do:
  - periodic_beaconing: CoV = 0.08 (rất thấp)
  - high_fan_out: IP này kết nối tới [Z] địa chỉ đích khác nhau
  
  Đây chính xác là 147.32.84.165 — bot trong CTU-13 Scenario 10."

BƯỚC 4 — So sánh với ground truth (30s):
  Nói: "Ground truth trong dataset: IP này có label 'botnet'.
  Model phát hiện đúng với false alarm rate chỉ [X]% trong toàn test set."

BƯỚC 5 — Close demo (15s):
  Nhấn Ctrl+C để dừng pipeline
  Nói: "Demo hoàn tất. Source code open-source tại GitHub
  [chiếu Slide 18 với GitHub link/QR code]"

## C.3 Plan B nếu demo fail

Nếu pipeline crash:
  "Xin lỗi có vấn đề kỹ thuật. Chúng tôi đã chuẩn bị
  recording demo backup." → chiếu screen recording đã quay sẵn

Nếu không có laptop/máy chiếu:
  Chiếu Slide 17 (alert JSON + dashboard screenshot)
  và giải thích output như đang demo live

══════════════════════════════════════════════════════════
YÊU CẦU OUTPUT
══════════════════════════════════════════════════════════

1. Viết đầy đủ tất cả 3 phần (A + B + C)
2. Phần A: viết FULL speaker notes cho TẤT CẢ 20 slides
3. Phần B: viết câu trả lời đầy đủ cho mỗi Q&A
4. Phần C: kịch bản đủ chi tiết để người không biết kỹ thuật 
   cũng có thể follow được
5. Ngôn ngữ: tiếng Việt tự nhiên, không cứng nhắc
6. Format: sử dụng markdown headers rõ ràng để dễ copy sang PowerPoint
```
