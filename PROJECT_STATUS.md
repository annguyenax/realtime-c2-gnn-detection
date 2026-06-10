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
- Verified metrics: **confirmed** — XGBoost F1=0.992, GraphSAGE F1=0.399 AUC=0.983
  FPR=0.09% (w=60s, class_weight_cap=50, filter_empty=True), GATv2 F1=0.052.

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

- ~~Run the full training pipeline to generate real artifacts.~~ **Done — F1=0.399.**
- Run threshold analysis: `python scripts/06_threshold_analysis.py --model graphsage --window-size 60`
- Add cross-scenario evaluation: train on scenario 10, test on scenario 8 or 11.
- Add window-size ablation: 30s, 120s (60s done, baseline 300s done).
- Record a short demo video showing preprocessing, training artifact, API, and dashboard.
- Write report chapters around honest limitations: CTU-13 age, C2 evasion, production scaling.
- (Optional) Replace StratifiedKFold with TimeSeriesSplit in XGBoost CV to prevent leakage.
