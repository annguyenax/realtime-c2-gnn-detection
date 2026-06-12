# Project Status and Defense Checklist

This file tracks the gaps that matter for grading, GitHub portfolio review, and
internship screening. Keep it honest: numbers are only final after the scripts
produce artifacts under `models/artifacts/` and `reports/`.

## Current State

- Data parser: implemented for CTU-13 `.binetflow` and parquet replay.
- Dynamic graph: implemented with sliding-window node and edge aggregation.
- Models: XGBoost, GraphSAGE, and GATv2 implementations are present.
- Realtime demo: implemented as a 3-thread replay pipeline.
- API/dashboard: implemented for alert ingestion and visualization.
- CI/security: GitHub Actions workflows exist for lint/test and Bandit/Trivy.
- Verified metrics: **confirmed** — XGBoost F1=0.992; GraphSAGE v3 (18-dim temporal):
  - Default thr=0.5: F1=0.3951, Prec=0.2675, Rec=0.7557, FPR=0.107%
  - **Tuned thr=0.9118: F1=0.6328, Prec=0.7106, Rec=0.5703, FPR=0.012%** ← primary result
  - AUC=0.9817, PR-AUC=0.6485, Latency=56.18ms/graph
  - Config: w=60s, class_weight_cap=50, filter_empty=True, temporal 18-dim features, seed=42
  - GATv2 F1=0.052 (untuned baseline).

## Session 6 Fixes (2026-06-11)

- `CosineAnnealingLR` T_max=100 hardcode fixed: `T_max = max(epochs*2, 100)` in `train()`
- Clean 70/15/15 temporal split: separate val and test, no overlap, `--train-ratio`/`--val-ratio` CLI flags
- `05_collect_metrics.py`: dual-threshold reporting (default + tuned), two rows per GNN model
- `data/README.md`: complete rewrite with CTU-13 citation, binetflow explanation, limitations
- `docs/questions_answers.md`: 25 Q&A for thesis defense created
- `docs/slides_outline.md`: 14-slide outline created
- `README.md`: updated with honest GraphSAGE v3 results (tuned F1=0.633)

## Fixed High-Priority Gaps

- GNN training script now imports the actual model classes.
- GNN training now uses `GNNTrainer.train()` instead of a missing `train_step`.
- Realtime graph snapshots are exported without ground-truth edge labels.
- Alert objects now carry a best-effort destination IP for API/dashboard context.
- Docker health checks can call `/health`.
- Dashboard model comparison reads verified metrics from `reports/final_metrics.json`.
- XGBoost metrics include normalized `precision`, `recall`, `f1`, and `pr_auc` keys.
- `GraphSAGEC2Detector` default `hidden_channels` matches training value (128).
- `_load_model` creates `GraphSAGEC2Detector(hidden_channels=128)` explicitly.
- `InferenceWorker._infer_and_alert` guards `edge_attr is None` before slicing.
- `GNNTrainer.benchmark_inference` strips ground-truth edge dim before forward pass.
- `05_collect_metrics.py` now includes `pr_auc` for graphsage and gatv2 entries.
- `03_train_xgboost.py` SHAP extraction handles `shap.Explanation` (SHAP >= 0.40).
- `dashboard/app.py` uses `df.style.map` (pandas >= 2.1 compatible, replaces `applymap`).
- `pipeline._default_handler` creates `reports/` directory before writing `alerts.jsonl`.
- Added `tests/test_gnn_integration.py`: snapshot → GNN forward pass, auto-skips without torch.

## Commands to Generate Evidence

```bash
python scripts/02_preprocess.py --scenario 10
python scripts/03_train_xgboost.py
python scripts/04_train_gnn.py --model graphsage --window-size 60 --max-class-weight 50 --filter-empty
python scripts/04_train_gnn.py --model gatv2 --window-size 60
python scripts/05_collect_metrics.py
python scripts/06_threshold_analysis.py --model graphsage --window-size 60
```

For a short smoke demo after training:

```bash
python -m c2gnn.realtime.pipeline \
  --data data/processed/scenario10_test.parquet \
  --model models/artifacts/graphsage_best.pt \
  --model-type graphsage \
  --max-flows 5000 \
  --realtime-factor 0
```

## Remaining Before a Strong Defense

- ~~Run the full training pipeline to generate real artifacts.~~ **Done — tuned F1=0.6328.**
- ~~CosineAnnealingLR T_max bug fixed.~~ **Done.**
- ~~Clean val/test split.~~ **Done — 70/15/15 temporal.**
- ~~`questions_answers.md` with 25 Q&A.~~ **Done.**
- Run `python scripts/05_collect_metrics.py` to regenerate `reports/final_metrics.json` with current numbers.
- Commit and push branch `feat/temporal-features-focal-loss` → merge to main.
- Smoke demo: `python -m c2gnn.realtime.pipeline --max-flows 2000 ...` with tuned threshold
- (Optional) Cross-scenario evaluation: train on Scenario 10, test on Scenario 8.
- (Optional) Window ablation: w=120s.
- Write report chapters around honest limitations: CTU-13 age, C2 evasion, production scaling.
- (Optional) Replace StratifiedKFold with TimeSeriesSplit in XGBoost CV.
