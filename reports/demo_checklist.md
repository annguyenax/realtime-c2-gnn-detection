# Checklist Bảo vệ — 95% Definition of Done

**Cập nhật:** 2026-06-10  
**Mục tiêu:** 95% hoàn thiện trước bảo vệ

---

## A. Code & Artifacts

| Done? | Hạng mục | Bằng chứng / Command |
|:---:|---|---|
| ✅ | CTU-13 parse và preprocess | `data/processed/scenario10_train.parquet` tồn tại |
| ✅ | Dataset stats | `data/processed/dataset_stats.json` tồn tại |
| ✅ | XGBoost train + metrics | `models/artifacts/xgboost_metrics.json` (F1=0.992) |
| ✅ | SHAP feature importance | `models/artifacts/shap_feature_importance.json` |
| ✅ | XGBoost classification report | `reports/xgboost_classification_report.txt` |
| ✅ | GraphSAGE train + metrics | `models/artifacts/graphsage_metrics.json` (F1=0.399, AUC=0.983, FPR=0.09%, w=60s, w_cap=50) |
| ✅ | GraphSAGE model checkpoint | `models/artifacts/graphsage_best.pt` |
| ✅ | final_metrics.json | `reports/final_metrics.json` ← sinh bởi script 05 |
| ✅ | results_table.txt | `reports/results_table.txt` |
| ⬜ | GATv2 train + metrics | `models/artifacts/gatv2_metrics.json` (cần chạy 04) |
| ⬜ | Threshold analysis JSON | `models/artifacts/graphsage_threshold_analysis.json` (chạy 06) |
| ⬜ | PR curve figure | `reports/figures/pr_curve_graphsage.png` (chạy 06) |
| ⬜ | Threshold sweep figure | `reports/figures/threshold_sweep_graphsage.png` (chạy 06) |
| ⬜ | Threshold sweep CSV | `reports/tables/threshold_sweep_graphsage.csv` (chạy 06) |
| ⬜ | Ablation window size | chạy 04 với --window-size 60, 120, 300 |
| ⬜ | Cross-scenario eval (Sc8) | tải Sc8, chạy 03 --gen-test |
| ⬜ | Confusion matrix figures | cần thêm vào script hoặc notebook |

---

## B. Real-time System

| Done? | Hạng mục | Bằng chứng / Command |
|:---:|---|---|
| ✅ | 3-thread pipeline chạy được | `python -m c2gnn.realtime.pipeline --help` |
| ✅ | Alert JSON format | xem README alert example |
| ✅ | FastAPI server | `src/c2gnn/api/server.py` |
| ✅ | Streamlit dashboard | `src/c2gnn/dashboard/app.py` |
| ✅ | Alert deduplication | implemented trong InferenceWorker |
| ⬜ | Smoke demo chạy được | `python -m c2gnn.realtime.pipeline --max-flows 2000 ...` |
| ⬜ | Demo video recorded | MP4 file trong `docs/` hoặc link |

---

## C. README & Repository

| Done? | Hạng mục | Bằng chứng |
|:---:|---|---|
| ✅ | README metrics đúng (không còn số giả) | `grep "0.930" README.md` → no match |
| ✅ | README có Limitations section | section ⚠️ Known Limitations |
| ✅ | README có Future Work | section 🔮 Future Work |
| ✅ | README định vị đúng "research prototype" | không claim "production-grade" |
| ✅ | File structure trong README khớp thật | verified |
| ✅ | .gitignore có mlflow.db, *.db | verified |
| ✅ | Repo không có cache bị track | `git ls-files \| grep __pycache__` → empty |
| ✅ | PROJECT_LOG.md tồn tại | file này |
| ⬜ | README có GIF demo hoặc screenshot | link hoặc file |
| ⬜ | `questions_answers.md` review — có nên public không? | xem nội dung |

---

## D. Báo cáo

| Done? | Hạng mục | File |
|:---:|---|---|
| ⬜ | Chương 1: Introduction | báo cáo draft |
| ⬜ | Chương 2: Problem Statement + Threat Model | |
| ⬜ | Chương 3: Related Work | |
| ⬜ | Chương 4: Dataset và Preprocessing | |
| ⬜ | Chương 5: Feature Engineering | |
| ⬜ | Chương 6: Dynamic Graph Construction | |
| ⬜ | Chương 7: Model Architecture | |
| ⬜ | Chương 8: Realtime System | |
| ⬜ | Chương 9: Experiments | |
| ⬜ | Chương 10: Results + Discussion | **phải honest về F1 thấp** |
| ⬜ | Chương 11: Limitations | **bắt buộc** |
| ⬜ | Chương 12: Future Work | |
| ⬜ | Chương 13: Conclusion | |
| ⬜ | Hình: Architecture diagram | |
| ⬜ | Hình: PR curve GNN | (chạy 06) |
| ⬜ | Hình: Threshold sweep | (chạy 06) |
| ⬜ | Hình: Confusion matrix XGBoost | |
| ⬜ | Hình: SHAP bar chart | |
| ⬜ | Bảng: Model comparison (số thật) | copy từ results_table.txt |
| ⬜ | Bảng: Dataset stats | copy từ dataset_stats.json |
| ⬜ | Bảng: Node feature list | từ NodeData.feature_names() |

---

## E. Slides (14 slides)

| Done? | Slide | Nội dung |
|:---:|---|---|
| ⬜ | 1 | Title, team, ngày |
| ⬜ | 2 | Vấn đề: C2 botnet detection gap |
| ⬜ | 3 | Threat model |
| ⬜ | 4 | Dataset: CTU-13 Scenario 10 |
| ⬜ | 5 | Dynamic graph construction |
| ⬜ | 6 | Node/Edge features (SHAP) |
| ⬜ | 7 | Model: XGBoost + GraphSAGE/GATv2 |
| ⬜ | 8 | XGBoost results (F1=0.992, port-dependent) |
| ⬜ | 9 | **GNN results + honest explanation** (threshold issue) |
| ⬜ | 10 | Threshold analysis (PR curve) |
| ⬜ | 11 | Realtime pipeline |
| ⬜ | 12 | Demo |
| ⬜ | 13 | Limitations + Future work |
| ⬜ | 14 | Kết luận |

---

## F. Câu hỏi phản biện — Chuẩn bị

| Done? | Câu hỏi | Trả lời sẵn? |
|:---:|---|---|
| ⬜ | Vì sao GNN F1 thấp? | AUC=0.949, threshold tuning, imbalance |
| ⬜ | Vì sao AUC cao nhưng F1 thấp? | AUC = ranking quality, F1 = threshold-dependent |
| ⬜ | Tại sao không chỉ dùng XGBoost? | Port-agnostic, neighborhood structure, inductive |
| ⬜ | CTU-13 lỗi thời không? | Yes, ghi limitation, still standard benchmark |
| ⬜ | Có data leakage không? | Temporal split, no leakage |
| ⬜ | Real-time thật không? | Replay simulation, "prototype" |
| ⬜ | Phát hiện encrypted C2 không? | Out of scope, ghi trong threat model |
| ⬜ | Scale lên SOC không? | NetworkX limit, Kafka future work |
| ⬜ | Node label hợp lý không? | Known limitation, any-adjacent strategy |
| ⬜ | Threshold chọn thế nào? | PR curve, validation-based |

---

## Commands để hoàn thiện nhanh

```bash
# Sinh threshold analysis và figures
python scripts/06_threshold_analysis.py --model graphsage

# Chạy GATv2 (nếu chưa có)
python scripts/04_train_gnn.py --model gatv2 --window-size 300

# Ablation window size
python scripts/04_train_gnn.py --model graphsage --window-size 60
python scripts/04_train_gnn.py --model graphsage --window-size 120

# Collect all metrics lại
python scripts/05_collect_metrics.py

# Smoke demo
python -m c2gnn.realtime.pipeline \
  --data data/processed/scenario10_test.parquet \
  --model models/artifacts/graphsage_best.pt \
  --model-type graphsage \
  --threshold 0.7 \
  --window-size 60.0 \
  --realtime-factor 0 \
  --max-flows 2000
```
