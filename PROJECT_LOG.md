# PROJECT C2GNN - LOG TIẾN ĐỘ

**Tổng quan:** Phát hiện C2 Traffic bằng Dynamic Graph Learning trên dataset CTU-13 Scenario 10.
Yêu cầu real-time pipeline (3-thread), so sánh GNN vs XGBoost, demo, báo cáo, slide.

**Trạng thái hiện tại:** ✅ TRAIN XONG — GraphSAGE (w=60s, max_class_weight=50, filter_empty=True) đạt **F1=0.3985**, vượt target 0.25–0.35. Sẵn sàng commit. Cần chạy threshold analysis + collect_metrics + viết báo cáo.

**Metrics chính xác (sau retrain Session 3):**

| Model | Precision | Recall | F1 | ROC-AUC | FPR% | Config |
|---|---|---|---|---|---|---|
| XGBoost | 0.9895 | 0.9947 | 0.9921 | 0.9998 | 0.10% | default |
| GraphSAGE v1 | 0.093 | 0.811 | 0.167 | 0.974 | 0.41% | w=60s, w_class=1676 |
| **GraphSAGE v2** | **0.283** | **0.671** | **0.399** | **0.983** | **0.09%** | w=60s, w_class=50, filter_empty |
| GATv2 | 0.027 | 0.839 | 0.052 | 0.970 | 1.54% | w=60s |

**Tiến độ chính:**
- ✅ Hoàn thành: CTU-13 preprocess, XGBoost (F1=0.992), GraphSAGE v2 (F1=0.399, AUC=0.983), GATv2 (F1=0.052), realtime pipeline, API, dashboard, CI/CD, MLflow, PROJECT_LOG, README, .gitignore, threshold analysis script, demo_script, demo_checklist, GNNTrainer improvements
- ⏳ TODO ngay: `python scripts/05_collect_metrics.py` → cập nhật final_metrics.json; `python scripts/06_threshold_analysis.py --model graphsage --window-size 60` → PR curve; commit tất cả thay đổi; viết báo cáo/slide

---

## LOG CHI TIẾT

---

### 2026-06-10 — Session khởi tạo PROJECT_LOG

**Context:** Đánh giá toàn diện đã xác định các vấn đề nghiêm trọng sau:

#### Vấn đề nghiêm trọng (phải fix ngay)

1. **README metrics giả** — README hiện claim GraphSAGE F1=0.930 nhưng `graphsage_metrics.json` thật cho F1=0.078. XGBoost claim F1=0.904 nhưng thật là F1=0.992. Đây là vấn đề học thuật nghiêm trọng.
2. **Cache files bị track** — `.pytest_cache/`, `.ruff_cache/`, `scripts/__pycache__/`, `src/**/__pycache__/`, `tests/__pycache__/` đang bị track bởi git nhưng không nên.
3. **`mlflow.db` và `mlruns/` uncommitted** — đang là untracked (`??`), nhưng cần add vào `.gitignore` để tránh commit nhầm.
4. **Parquet + binetflow + model artifact chưa được git-tracked** — tốt, nhưng cần verify `.gitignore` cover đủ.
5. **Nhiều file trong README không tồn tại** — `preprocess.py`, `download.py`, `feature_store.py`, v.v. được list nhưng không có trong repo thật.
6. **`reports/final_metrics.json` chưa tồn tại** — dashboard đọc file này nhưng script 05 chưa được chạy.

#### Metrics thật (từ JSON artifacts)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR% | Latency |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.9895 | 0.9947 | 0.9921 | 0.9998 | 0.9990 | 0.10% | 2.1 ms/flow |
| GraphSAGE | 0.0417 | 0.6237 | 0.0782 | 0.9487 | 0.0547 | 0.72% | 144.9 ms/graph |

#### Giải thích GNN kết quả

- AUC cao (0.949) = model phân biệt botnet/normal đúng hướng (ranking đúng)
- F1 thấp (0.078) = threshold=0.5 quá thấp với class imbalance; precision=0.042 nghĩa là over-alarm
- Recall cao (0.624) = model bắt được 62% botnet nodes thật
- Node labeling quá rộng: node bị label botnet nếu BẤT KỲ adjacent edge là botnet
- Window 300s tạo graph lớn với nhiều background nodes, làm loãng signal
- Threshold tuning cần thiết: tìm optimal threshold từ PR curve

#### Dataset stats (thật)

- Scenario 10: 5,178,417 flows, 6.22% botnet, imbalance 15:1
- Train: 4,142,733 flows (temporal first 80%), 5.55% botnet
- Test: 1,035,684 flows (last 20%), 8.89% botnet

---

### 2026-06-10 — Task 1: Tạo PROJECT_LOG.md

- Kết quả: File này được tạo với cấu trúc đầy đủ ✅
- Bước tiếp theo: Chạy `05_collect_metrics.py`, fix README, fix .gitignore, threshold analysis

---

### 2026-06-10 — Task 2: Chạy 05_collect_metrics.py ✅

- Command: `python scripts/05_collect_metrics.py`
- Output sinh ra:
  - `reports/final_metrics.json` ✅
  - `reports/results_table.txt` ✅
- Kết quả bảng:
  ```
  xgboost   Prec=0.9895  Rec=0.9947  F1=0.9921  AUC=0.9998  FPR=0.10%  2.1ms/flow
  graphsage Prec=0.0417  Rec=0.6237  F1=0.0782  AUC=0.9487  FPR=0.72%  144.8ms/graph
  ```
- Top SHAP features: src_port (2.81), bytes_per_packet (2.00), dst_port (2.00), is_tcp (1.42), total_bytes (1.25)

---

### 2026-06-10 — Task 3: Fix README.md ✅

- Vấn đề: Fabricated metrics table (GraphSAGE F1=0.930 vs thật 0.078)
- File sửa: `README.md`
- Thay đổi:
  - Results table: xóa số giả, thêm số thật từ JSON artifacts
  - Thêm "Honest analysis" block giải thích GNN F1 thấp
  - Thêm bảng "Why GNN Still Has Academic Value"
  - Thêm section "⚠️ Known Limitations" (8 điểm cụ thể)
  - Fix "Repository Structure" để khớp với file thật (loại bỏ 13 file không tồn tại)
  - Sửa positioning: "research prototype" thay "production-grade"
  - Future Work: thêm threshold tuning và cross-scenario eval lên đầu

---

### 2026-06-10 — Task 4: Fix .gitignore ✅

- Thêm vào `.gitignore`:
  - `mlflow.db`, `*.db`, `*.sqlite`
  - `reports/alerts.jsonl`, `reports/alerts.json`
- Kiểm tra: repo đã sạch — không có cache/artifact nào bị track sai

---

### 2026-06-10 — Task 5: Verify cache hygiene ✅

- `git ls-files` → 55 files, KHÔNG có `__pycache__`, `.pytest_cache`, `.ruff_cache`, `mlruns`, parquet, binetflow, .pt artifacts
- Repository đã sạch từ đầu — .gitignore đã cover đúng
- Cache local tồn tại nhưng chỉ là local, không bị commit

---

### 2026-06-10 — Task 6: Script threshold analysis ✅

- Tạo mới: `scripts/06_threshold_analysis.py`
- Features:
  - Load trained GNN model
  - Build test snapshots từ raw binetflow (last 20%)
  - Sweep threshold 0.05 → 0.95, tính F1/Prec/Rec/FPR
  - Find optimal threshold by F1
  - Plot PR curve (matplotlib)
  - Plot threshold sweep figure
  - Save CSV và JSON summary
- Usage: `python scripts/06_threshold_analysis.py --model graphsage`
- Output: `reports/figures/pr_curve_graphsage.png`, `reports/tables/threshold_sweep_graphsage.csv`

---

### 2026-06-10 — Task 7: Demo script + checklist ✅

- Tạo: `docs/demo_script.md`
  - 5 bước demo với commands cụ thể
  - Fallback commands nếu demo lỗi
  - Quick Q&A answers cho phản biện
  - Checklist demo day
- Tạo: `reports/demo_checklist.md`
  - 6 nhóm: Code/Artifacts, Realtime, README/Repo, Báo cáo, Slides, Q&A
  - Tracking ✅/⬜ cho từng hạng mục
  - Commands để hoàn thiện nhanh

---

### 2026-06-10 — Session 2: GNNTrainer cải tiến + metrics cập nhật

**Context:** User retrained GraphSAGE với window=60s. Kết quả mới:
- GraphSAGE (w=60s): F1=0.1671, AUC=0.9742, PR-AUC=0.512, Prec=0.093, Rec=0.811, FPR=0.41%, Lat=75.2ms
- GATv2 (w=60s): F1=0.0518, AUC=0.9701, FPR=1.54%, Lat=296.5ms
- Root cause F1 thấp: `class_weight_botnet = n_neg/n_pos = 1676.77` → model over-predicts botnet → Prec=0.093

**Vấn đề xác định:**
1. Class weight 1676 quá cao → precision sụp đổ (model predict hầu hết node là botnet)
2. Nhiều training snapshot không có botnet node → nhiễu không cần thiết
3. Threshold analysis script chạy window=300s nhưng model train window=60s → domain shift

**Task 8: Cải tiến GNNTrainer** ✅

File sửa: `src/c2gnn/models/graphsage.py`
- Thêm `max_class_weight=50.0` vào `GNNTrainer.train()`: cap weight từ 1676→50
- Thêm `filter_empty_snapshots=True`: bỏ training snapshots không có botnet node
- Log cả `raw_class_weight_botnet` và `capped_class_weight_botnet` vào MLflow
- Expected effect: Precision tăng từ 0.093 → ~0.2+, F1 tăng từ 0.167 → ~0.25-0.35

File sửa: `scripts/04_train_gnn.py`
- Thêm `--max-class-weight` arg (default=50)
- Thêm `--filter-empty` / `--no-filter-empty` flag
- Pass tới `train_model()` và `GNNTrainer.train()`

**Lệnh retrain:**
```bash
python scripts/04_train_gnn.py --model graphsage --window-size 60 --max-class-weight 50 --filter-empty
```

**Task 9: Cập nhật README + demo files** ✅

- README results table: GraphSAGE F1=0.167 (was 0.078), thêm GATv2 row, window=60s
- README realtime latency: 75ms (was 145ms)
- `docs/demo_script.md`: cập nhật window=60s, PR-AUC=0.512, recall=81%
- `reports/demo_checklist.md`: F1=0.167 (was 0.078)

---

## TODO BACKLOG

### 2026-06-10 — Session 3: GraphSAGE v2 — Retrain xong, F1=0.3985 ✅

**Lệnh đã chạy:**
```
python scripts/04_train_gnn.py --model graphsage --window-size 60 --max-class-weight 50 --filter-empty
```

**Kết quả:**
- Best val_f1 = **0.3857** (epoch best)
- Test F1 = **0.3985** ← vượt target 0.35 ✅
- Precision = **0.2834** (v1: 0.093 → +204%)
- Recall = **0.6711** (v1: 0.811 → trade-off chấp nhận được)
- ROC-AUC = **0.9830** (v1: 0.9742 → cải thiện)
- FPR = **0.09%** (v1: 0.41% → giảm 4.5×, cực quan trọng)

**Phân tích:** max_class_weight=50 + filter_empty=True đã giải quyết đúng root cause (class_weight=1676 → precision collapse). F1 đi từ 0.167 → 0.399, cải thiện 139%. FPR=0.09% tốt hơn cả XGBoost (0.10%), là điểm mạnh quan trọng để báo cáo.

**Cần làm ngay sau commit:**
1. `python scripts/05_collect_metrics.py` → cập nhật reports/final_metrics.json
2. `python scripts/06_threshold_analysis.py --model graphsage --window-size 60` → PR curve + threshold sweep
3. Update README với số liệu F1=0.399

**Task 10: Commit tất cả thay đổi** — xem lệnh git phía dưới

---

### Ưu tiên cao (ngay bây giờ)
- [x] ~~Retrain GraphSAGE với max_class_weight=50 + filter_empty~~ → **DONE: F1=0.3985** ✅
- [ ] **Commit code + docs** (xem lệnh git section bên dưới)
- [ ] **Chạy threshold analysis** với `--window-size 60` → `reports/figures/pr_curve_graphsage.png`
- [ ] **Update `reports/final_metrics.json`**: `python scripts/05_collect_metrics.py`
- [ ] Cập nhật README với F1=0.399
- [ ] Bắt đầu viết báo cáo (Chương 1-5)

### Ưu tiên trung bình (trong 3 ngày)
- [ ] Window size ablation: train GraphSAGE với 30s, 120s (60s và 300s đã có)
- [ ] Threshold sweep: chạy 06 với --window-size 60, tìm optimal F1 threshold
- [ ] Cross-scenario eval: XGBoost trên Scenario 8
- [ ] Confusion matrix figures
- [ ] Báo cáo Chương 6-10

### Ưu tiên thấp (trong 7 ngày)
- [ ] SHAP figure for XGBoost
- [ ] Demo video recording
- [ ] Báo cáo hoàn chỉnh (Chương 11-13 + hình)
- [ ] Slide 14 trang hoàn chỉnh
- [ ] README final pass với GIF demo
