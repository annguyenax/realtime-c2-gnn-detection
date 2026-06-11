# Questions & Answers — Phản biện bảo vệ đồ án

**Đề tài:** Phát hiện C2 Traffic bằng Dynamic Graph Learning (Realtime C2 Detection via Dynamic Graph Neural Networks)

---

## Phần 1: Dataset và preprocessing

**Q1. Dataset CTU-13 lấy từ đâu? Nguồn chính thức là gì?**

CTU-13 được thu thập tại Czech Technical University (CTU), Prague, Czech Republic. Nguồn mô tả chính thức là trang Stratosphere IPS Lab: https://www.stratosphereips.org/datasets-ctu13. Nguồn tải dữ liệu authoritative là MCFP server: https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/. Citation chuẩn: Garcia et al., *An empirical comparison of botnet detection methods*, Computers & Security, 2014.

**Q2. CTU-13 có phải định dạng NetFlow không?**

Không hoàn toàn. CTU-13 dùng **bidirectional Argus flows** (`.binetflow`), không phải NetFlow v5/v9 tiêu chuẩn. Argus gộp hai chiều của một kết nối vào một dòng duy nhất, trong khi NetFlow thường unidirectional. Điều này ảnh hưởng đến cách đọc bytes/packets — đây là lưu ý khi so sánh với các dataset khác.

**Q3. CTU-13 có lỗi thời không?**

Có. Dataset được thu thập năm 2011. C2 hiện đại (2020+) dùng HTTPS/TLS, domain fronting, DGA, Cobalt Strike qua Azure — hoàn toàn khác với IRC-based botnet trong CTU-13. Dự án định vị là **proof-of-concept nghiên cứu** trên dataset chuẩn của cộng đồng, không claim phát hiện C2 hiện đại trong môi trường production.

**Q4. Tại sao chỉ dùng Scenario 10?**

Scenario 10 (Murlo botnet, IRC C2) được chọn vì: (1) là scenario phổ biến nhất trong literature, (2) có đủ số flows (5.17M) để xây dựng graph snapshots có ý nghĩa thống kê, (3) có beaconing rõ ràng (IRC với khoảng cách đều đặn) phù hợp để test temporal features. Cross-scenario evaluation (Scenario 8, Rbot) được liệt kê là hướng phát triển tương lai.

**Q5. Train/test split có bị data leakage không?**

Split được thực hiện **theo thời gian** (temporal split) — không shuffle ngẫu nhiên. 70% đầu tiên là train, 15% tiếp theo là validation (dùng để tune threshold), 15% cuối là test. Không có flow nào từ tương lai xuất hiện trong training. Threshold tuning chỉ dùng validation set, không dùng test set.

---

## Phần 2: XGBoost và so sánh model

**Q6. Tại sao XGBoost F1 cao hơn GNN nhiều vậy (0.992 vs 0.633)?**

Có hai lý do chính. Thứ nhất, XGBoost là **flow-level tabular classifier** — nó phân loại từng flow độc lập dựa trên 14 đặc trưng số học như src_port, dst_port, bytes, packets. CTU-13 Scenario 10 (Murlo botnet) dùng IRC qua port 6667 — đây là feature cực kỳ discriminative. SHAP analysis cho thấy `src_port` và `dst_port` là hai features quan trọng nhất (SHAP > 2.0). Thứ hai, GNN hoạt động ở **graph/node level** — nó nhìn vào cấu trúc topology và các quan hệ giữa IPs trong một time window, không phải từng flow riêng lẻ. Nếu botnet đổi port, XGBoost sẽ thất bại, còn GNN vẫn nhận ra pattern qua temporal graph features.

**Q7. Tại sao XGBoost train nhanh hơn GNN?**

XGBoost train trên tabular data: mỗi sample là một vector đặc trưng, không cần xây dựng graph. Với 5M flows, XGBoost mất vài giây. GNN phải: (1) parse flows theo thứ tự thời gian, (2) xây dựng sliding window graph cho mỗi snapshot (~60s), (3) thực hiện message passing qua nhiều lớp SAGE convolution, (4) train trên 421 snapshots mỗi epoch × 50 epochs. Tổng thời gian train GNN: ~40-90 phút (CPU).

**Q8. XGBoost có leakage do port feature không?**

Đây là limitation được thừa nhận rõ. XGBoost đạt F1=0.992 **largely because Neris/Murlo botnet uses IRC port 6667** — đây là signature-based detection, không phải behavioral detection. Nếu thay port hoặc dùng botnet khác, XGBoost sẽ bị degraded. Để kiểm chứng, cần cross-scenario evaluation (train on Sc.10, test on Sc.8 with Rbot). GNN, ngược lại, học từ graph structure và temporal patterns — port-agnostic hơn.

**Q9. XGBoost có dùng time split không?**

Có. XGBoost cũng dùng temporal split (không shuffle) để đảm bảo tính nhất quán với GNN evaluation. Cross-validation được dùng chỉ để estimate variance, không làm split cuối cùng.

---

## Phần 3: GNN — F1 thấp, AUC cao

**Q10. Tại sao GNN F1 mặc định thấp (0.395) nhưng AUC lại cao (0.982)?**

Đây là hiện tượng phổ biến trong bài toán **class imbalance node classification**. AUC (ROC-AUC) đo khả năng **ranking** của model — nó cho biết model có đặt botnet nodes lên trên normal nodes không, và câu trả lời là Có (AUC=0.982 rất tốt). F1 tại threshold=0.5 lại phụ thuộc vào **calibration** của xác suất đầu ra. Với class imbalance nặng (~99% node là normal), model học xu hướng predict xác suất thấp cho tất cả nodes — ngay cả botnet nodes cũng bị predict < 0.5. Kết quả: F1 mặc định thấp dù model ranking rất tốt.

**Q11. Threshold tuning có hợp lệ về mặt phương pháp không?**

Có, hoàn toàn hợp lệ. Threshold tuning là bước chuẩn trong machine learning cho bài toán imbalanced classification. Quan trọng là threshold phải được tìm trên **validation set** (không phải test set). Trong project này: threshold được tìm bằng cách tối đa hóa F1 trên validation set với constraint min_recall=0.40. Sau đó threshold này được apply lên test set. Đây không phải là information leakage.

**Q12. F1_tuned=0.633 có đáng tin không? Có overfit threshold không?**

F1_tuned được tính trên **test set** — tập hoàn toàn tách biệt với validation set dùng để tìm threshold. Không có overfit. Threshold=0.9118 được tìm trên validation (15% data), sau đó apply lên test (15% data cuối). Test set chưa từng được dùng trong quá trình training hay threshold tuning.

**Q13. Tại sao cần threshold cao vậy (0.9118)? Có bình thường không?**

Threshold cao là hệ quả của **model calibration với class imbalance**. Model được train với class weight ratio ~50:1, điều này khiến output xác suất bị shift xuống. Botnet nodes thường có xác suất P(botnet) trong khoảng 0.5–0.95, không phải 0.95–1.0 như người ta thường nghĩ. Threshold=0.9118 nghĩa là: chỉ alert khi model rất tự tin, giúp tăng precision lên 71% và giảm FPR xuống 0.012%.

**Q14. FPR=0.012% có nghĩa gì trong thực tế?**

Trong mạng doanh nghiệp với 1 triệu nodes/ngày, FPR=0.012% = 120 false alarms mỗi ngày. Đây vẫn là con số đáng kể cho SOC analyst. Tuy nhiên so với baseline GATv2 (FPR=1.54% = 15,400 false alarms) thì GraphSAGE tốt hơn 128 lần. Post-processing (k/n alert smoothing) có thể giảm FPR thêm mà không cần retrain.

---

## Phần 4: Graph và temporal features

**Q15. Tại sao dùng graph learning thay vì chỉ dùng flow-level ML?**

Graph learning capture được **relational context** mà flow-level ML bỏ qua. Ví dụ: một IP bình thường sẽ giao tiếp đa dạng với nhiều IPs khác nhau. Một bot IP, ngược lại, sẽ kết nối định kỳ đến cùng một C2 server với inter-arrival time đều đặn (`iat_cv` thấp) và `repeat_dst_ratio` cao. Graph structure cho phép phát hiện các pattern này ở mức topology — không phải chỉ dựa vào từng flow.

**Q16. Temporal features (iat_cv, repeat_dst_ratio) đóng góp gì?**

Temporal features là điểm khác biệt chính của GNN v3 so với v2. `iat_cv` (coefficient of variation of inter-arrival times) đặc biệt quan trọng: C2 beaconing có `iat_cv ≈ 0.05–0.25` (rất đều đặn), trong khi traffic bình thường có `iat_cv > 1.0` (ngẫu nhiên). Thêm 4 temporal features đã cải thiện tuned F1 từ ~0.45 lên ~0.63.

**Q17. Node labeling có hợp lý không? Có over-label không?**

Một node (IP) được label `botnet=1` nếu **bất kỳ edge** nào liên kết với node đó trong time window là botnet flow. Điều này có thể **over-label hub nodes** như DNS servers, gateways nếu chúng tình cờ communicate với botnet IP. Đây là limitation được thừa nhận. Giải pháp chính xác hơn là label node theo tỷ lệ botnet edges hoặc chỉ label src IP của botnet flows. Để không thay đổi label schema giữa chừng, limitation này được document trong report.

---

## Phần 5: Real-time system

**Q18. Realtime thật hay giả lập?**

Hệ thống có 3-thread pipeline thật sự: **FlowBuilderWorker** đọc flows theo thứ tự thời gian và có thể điều chỉnh tốc độ replay bằng `--realtime-factor`. Với `--realtime-factor 50`, 50 flows/giây được xử lý (nhanh hơn thực tế 50 lần). Đây là **controlled replay** của traffic thật, không phải giả lập ngẫu nhiên. Trong production, FlowBuilderWorker sẽ nhận flows từ network tap hoặc NetFlow collector.

**Q19. Latency 56ms/graph có ý nghĩa gì? Có đủ cho realtime không?**

56ms là inference time cho một graph snapshot (tất cả nodes trong 60s window, trên CPU). Với pipeline snapshot mỗi 30s (50% overlap), system cần xử lý mỗi snapshot trong < 30,000ms — còn rất nhiều margin. Ngay cả với 100ms/snapshot, latency vẫn ổn. Detection latency thực tế từ flow xuất hiện đến alert là: flow timestamp + window accumulation (0-60s) + inference (56ms) + alert propagation ≈ 30–60 giây. Đây là **near-realtime**, không phải sub-second detection.

**Q20. Có scale lên môi trường production không?**

Hiện tại không, vì một số lý do: (1) Graph backend là NetworkX (single-threaded Python, không scale đến >10k concurrent nodes). Cần thay bằng distributed graph store (GraphBolt, DGL distributed). (2) Model chạy trên CPU. Với CUDA, latency giảm 10-50×. (3) Pipeline chưa có persistence layer cho graph state. (4) Alert API chưa có authentication/authorization. Dự án định vị là **research prototype và demo**.

---

## Phần 6: Methodology và tổng quan

**Q21. Tại sao không so sánh với các GNN khác (GCN, GraphSAINT...)?**

GATv2 đã được implement và so sánh. GCN không phù hợp vì non-inductive (không handle unseen nodes — bất kỳ IP mới nào trong production). GraphSAINT và GraphBolt phù hợp cho large-scale graphs nhưng phức tạp hơn cần thiết với dataset này. GraphSAGE được chọn vì inductive, scalable, và được sử dụng rộng rãi trong network security literature.

**Q22. GNN có thể phát hiện C2 dùng DGA hoặc domain fronting không?**

DGA (Domain Generation Algorithm): DGA C2 tạo nhiều domain ngẫu nhiên nhưng vẫn beaconing đều đặn đến C2. `iat_cv` thấp vẫn là signal. Tuy nhiên nếu traffic được tunnel qua HTTPS, flow-level features sẽ ít discriminative. Domain fronting và CDN-based C2 (Cobalt Strike qua Cloudflare): hiện tại không phát hiện được — đây là limitation rõ ràng.

**Q23. Giá trị học thuật của dự án là gì?**

1. Implement và validate complete pipeline từ raw NetFlow đến real-time GNN alert.
2. So sánh graph-based (GNN) vs tabular (XGBoost) trên cùng dataset và evaluation protocol.
3. Phân tích cold-start gap: warm-start F1=0.652 vs cold-start F1=0.060 — finding quan trọng cho deployment strategy.
4. Temporal beaconing features (iat_cv, repeat_dst_ratio) đóng góp rõ ràng: F1 tăng từ 0.399 lên 0.633.
5. Document limitation trung thực: không overclaim, phân tích rõ tại sao XGBoost mạnh hơn trên CTU-13.

**Q24. Threshold tuning trên validation set có nên được report riêng không?**

Có và đây là best practice. Report phải rõ ràng phân biệt:
- **Default threshold (0.5):** F1=0.395, Prec=0.268, Rec=0.756 — đây là kết quả "raw" không optimize
- **Tuned threshold (0.9118):** F1=0.633, Prec=0.711, Rec=0.570 — đây là kết quả thực tế khi deploy

Không nên chỉ report F1=0.395 vì gây hiểu nhầm model kém. Không nên chỉ report F1=0.633 mà không giải thích threshold — gây hiểu nhầm quá trình optimize không minh bạch.

**Q25. Điểm yếu nhất của dự án là gì và hướng cải thiện?**

Điểm yếu chính:
1. **Single-scenario training** — chưa validate cross-scenario generalization
2. **Cold-start dependency** — model cần warm-up graph history để hoạt động tốt
3. **Port-dependent XGBoost** — high F1 không generalizable nếu C2 đổi port
4. **Dataset age** — CTU-13 (2011) không đại diện cho C2 hiện đại

Hướng cải thiện:
1. Cross-scenario evaluation (Sc.10 train → Sc.8 test)
2. Warm graph initialization từ normal traffic profile trước deployment
3. Post-processing k/n alert smoothing để tăng precision
4. Thử với dataset mới hơn (CICIDS2017, UNSW-NB15, CIC-IDS2018)
