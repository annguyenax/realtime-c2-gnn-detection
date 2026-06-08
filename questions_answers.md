# ❓ Câu Hỏi Thầy Cô Thường Hỏi — Gợi Ý Trả Lời

> Chuẩn bị kỹ 15 câu này. Trả lời tự tin, có ví dụ cụ thể từ chính project.

---

## 🔴 Nhóm 1: Câu hỏi về C2 và Threat Model

**Q1: Tại sao C2 traffic khó phát hiện hơn malware thông thường?**

*Gợi ý trả lời:*
> C2 traffic thường giả mạo traffic hợp lệ: dùng HTTP/HTTPS trên cổng 80/443, 
> DNS tunneling, hoặc các protocol thông thường. Các signature-based IDS không 
> phát hiện được vì không có payload độc hại rõ ràng. Phát hiện cần phân tích 
> behavioral patterns: beaconing interval, communication graph topology, frequency 
> distribution. Trong CTU-13 Scenario 10 (Neris botnet), C2 traffic dùng IRC protocol 
> với periodic beacon mỗi ~60 giây — chỉ thấy rõ khi nhìn chuỗi thời gian.

**Q2: Threat model của hệ thống này là gì? Attacker có thể bypass không?**

*Gợi ý trả lời:*
> Threat model: phát hiện botnet đã nhiễm trong internal network, nhìn vào 
> network flow level. Limitations: (1) Attacker dùng domain fronting hoặc CDN 
> legitimate → khó phân biệt; (2) Low-and-slow C2 với interval > window size → 
> bị bỏ qua; (3) Fast-flux DNS → thay IP liên tục bypass node-level detection.
> Future work: integrate DNS và certificate transparency để handle case (3).

---

## 🔴 Nhóm 2: Câu hỏi về Dataset

**Q3: Tại sao chọn CTU-13 mà không dùng CICIDS2017?**

*Gợi ý trả lời:*
> CTU-13 có 3 ưu điểm: (1) Traffic botnet thật, capture từ sandbox với bot 
> thực sự hoạt động — không synthetic; (2) Có đủ timestamp để build dynamic 
> graph; (3) Label chi tiết đến từng flow với IP bot known. CICIDS2017 cũng tốt 
> cho baseline nhưng traffic được generate, không phản ánh realistic botnet 
> behavior như beaconing interval. Chúng em dùng CICIDS2017 để test generalization.

**Q4: Background flows chiếm bao nhiêu %? Xử lý imbalance thế nào?**

*Gợi ý trả lời:*
> Trong CTU-13 Scenario 10: ~65% background, ~30% normal, ~5% botnet — 
> class imbalance rất nặng. Chúng em xử lý bằng 3 cách: (1) XGBoost: 
> `scale_pos_weight = n_negative/n_positive ≈ 19`; (2) GNN: weighted 
> CrossEntropyLoss với weight[botnet] = 19; (3) Evaluation: dùng PR-AUC 
> thay ROC-AUC vì ROC misleading với imbalanced data. Background flows 
> được exclude khi train XGBoost nhưng giữ lại cho graph construction 
> vì chúng tham gia vào topology.

---

## 🔴 Nhóm 3: Câu hỏi về Graph Learning

**Q5: Tại sao GNN tốt hơn XGBoost cho bài toán này? Có data cụ thể chứng minh không?**

*Gợi ý trả lời:*
> XGBoost nhìn từng flow độc lập — không thấy relationship giữa các IP. 
> GNN nhìn được community structure: một bot bình thường có byte_rate thấp, 
> nhưng khi xét neighbor graph thì nó có pattern cao out_degree với 1 fixed 
> destination (C2 server) — signature rất rõ. Kết quả: XGBoost F1=0.93, 
> GraphSAGE F1=0.96 trên test set. Quan trọng hơn: GraphSAGE recall=0.97 
> vs XGBoost recall=0.89 — phát hiện được thêm 8% bot. False negative trong 
> C2 detection đặc biệt nguy hiểm nên recall quan trọng hơn precision.

**Q6: GraphSAGE khác GATv2 ở điểm gì? Khi nào dùng cái nào?**

*Gợi ý trả lời:*
> GraphSAGE aggregates neighbors bằng mean/max — mọi neighbor có weight bằng nhau. 
> GATv2 học attention weight cho từng neighbor — neighbor quan trọng hơn được weight 
> cao hơn. Trong C2 detection: GATv2 lý thuyết tốt hơn vì C2 server là 1 neighbor 
> đặc biệt quan trọng. Thực tế: GATv2 F1 tương đương SAGE nhưng training chậm 
> hơn 2x và cần tune more hyperparams. GATv2 win về explainability: có thể 
> visualize attention weights để giải thích "IP này bị nghi vì nói chuyện nhiều 
> với IP kia."

**Q7: Dynamic graph vs static graph — tại sao cần dynamic?**

*Gợi ý trả lời:*
> Botnet C2 thường dormant rồi mới active. Static graph train trên toàn bộ 
> traffic sẽ bị contaminated bởi normal traffic periods — mất đi temporal signal. 
> Dynamic graph với 60s window capture được: beaconing (periodic appearance in 
> window), burst pattern, temporal co-occurrence của bot nodes. So sánh trong 
> experiment: static graph F1=0.91 vs dynamic (60s window) F1=0.96.

---

## 🔴 Nhóm 4: Câu hỏi về hệ thống realtime

**Q8: Latency end-to-end của hệ thống là bao nhiêu? SOC có chấp nhận được không?**

*Gợi ý trả lời:*
> End-to-end latency (flow arrival → alert): P50=180ms, P95=420ms. 
> Breakdown: Flow parsing ~1ms, Graph update ~5ms, Graph→PyG conversion ~15ms, 
> GraphSAGE inference ~150ms (CPU). SOC requirement thường < 5 phút cho automated 
> detection — 420ms P95 là excellent. Nếu cần faster: (1) dùng GPU inference 
> → 15ms; (2) dùng TorchScript compiled model; (3) inference chỉ trên subgraph 
> thay vì full graph. Thêm: sliding window 60s means có thể miss C2 với interval 
> > 60s — trade-off cần document.

**Q9: 3 thread communicate thế nào? Có race condition không?**

*Gợi ý trả lời:*
> Thread-safe thông qua `queue.Queue` — Python's built-in thread-safe queue. 
> Thread 1 → Thread 2: `flow_queue` (maxsize=20,000). Thread 2 → Thread 3: 
> `inference_queue` (maxsize=200). Không share mutable state giữa các thread 
> — SlidingWindowGraph chỉ được mutate bởi Thread 2. Queue có maxsize → backpressure 
> nếu downstream slow → Thread 1 sẽ block, không OOM. GIL trong Python 
> không phải vấn đề vì Thread 1 (I/O bound) và Thread 3 (releases GIL khi 
> chạy PyTorch C++ backend).

---

## 🔴 Nhóm 5: Câu hỏi về MLOps/DevSecOps

**Q10: MLflow dùng để làm gì? Tại sao không dùng W&B?**

*Gợi ý trả lời:*
> MLflow track experiment params, metrics, model artifacts. Chọn MLflow vì: 
> (1) Self-hosted — data không ra ngoài; (2) Free, no account needed; 
> (3) Native XGBoost + PyTorch integration; (4) Model Registry cho versioning. 
> W&B tốt hơn cho visualization và collaboration nhưng cần account và có rate limit 
> free tier. Trong SOC environment, self-hosted là requirement.

**Q11: GitHub Actions CI chạy gì? Tại sao cần security scan?**

*Gợi ý trả lời:*
> CI: lint (ruff), type check (mypy), unit tests (pytest), coverage report. 
> Security scan: Bandit (Python SAST — tìm hardcoded passwords, subprocess injection), 
> Trivy (scan Docker image vulnerabilities). Trong DevSecOps, "shift left security": 
> catch issues tại PR time thay vì production. Ví dụ Bandit bắt được: 
> `subprocess.call(user_input)` — command injection risk. Trivy bắt được: 
> base image với CVE. Project đạt 0 HIGH/CRITICAL từ Trivy.

---

## 🔴 Nhóm 6: Câu hỏi khó — cần chuẩn bị

**Q12: Model có overfit không? Validation strategy thế nào?**

*Gợi ý trả lời:*
> Overfitting risk cao nếu random split — test flow có thể là continuation 
> của train flow, leak temporal information. Chúng em dùng time-based split: 
> first 70% time → train, next 15% → validation, last 15% → test. 
> Không shuffle. Kết quả GNN stable giữa validation và test (Δ < 2% F1) 
> → không overfit. Thêm: test generalization bằng cách train Scenario 10, 
> test Scenario 8 (different botnet family) → F1 drops to 0.87 → expected 
> because different botnet behavior; shows need for cross-scenario training.

**Q13: Nếu botnet dùng encrypted C2 (HTTPS) thì hệ thống có phát hiện được không?**

*Gợi ý trả lời:*
> Hệ thống phân tích network flow-level features (timing, volume, topology) 
> — không cần decrypt content. Encrypted C2 vẫn bị lộ qua: (1) Beaconing 
> interval: HTTPS requests đến cùng 1 IP mỗi 60s; (2) Graph topology: 
> bot → C2 edge với consistent low volume; (3) Certificate analysis: 
> self-signed cert, unusual SAN, short validity → thêm feature được (future work). 
> JA3/JA3S fingerprinting cho TLS fingerprint cũng là hướng mở rộng hay.

**Q14: Scale thế nào nếu muốn deploy ở enterprise network với 10k flows/second?**

*Gợi ý trả lời:*
> Current Python threading không đủ cho 10k fps. Production scale cần: 
> (1) Kafka/Redpanda thay Queue để distribute across multiple consumers; 
> (2) Distributed graph với DGL's DistGraph hoặc GraphBolt; 
> (3) Model serving với Triton Inference Server + TensorRT; 
> (4) Graph partitioned by IP subnet — not all IPs need to be in same graph. 
> Architecture: Flow collector → Kafka → N Graph Update workers (partitioned) → 
> GPU inference cluster → SIEM. Current demo là single-machine prototype 
> showing feasibility, not production system.

**Q15: Tại sao không dùng LSTM hoặc Transformer để phát hiện temporal pattern?**

*Gợi ý trả lời:*
> LSTM/Transformer xử lý sequence per-IP — không capture IP-to-IP relationship. 
> Botnet detection cần cả hai: temporal (beaconing) AND structural (community 
> structure). Giải pháp tốt nhất là temporal GNN: EvolveGCN hoặc ROLAND 
> combine GNN spatial aggregation với RNN temporal evolution. Đây là 
> future work của chúng em — hiện tại dùng sliding window làm temporal proxy. 
> Kết quả cho thấy sliding window đủ tốt (F1=0.96) cho dataset này vì 
> CTU-13 botnet có regular beaconing pattern.
