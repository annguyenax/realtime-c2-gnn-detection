"""
Train XGBoost C2 detector on CTU-13 Scenario 10.

Outputs:
  models/artifacts/xgboost_model.json        (trained model)
  models/artifacts/xgboost_metrics.json      (F1, AUC, FPR, latency)
  models/artifacts/shap_feature_importance.json
  reports/xgboost_classification_report.txt

Also logs everything to MLflow (run: mlflow ui --port 5001)

Usage:
    python scripts/03_train_xgboost.py
    python scripts/03_train_xgboost.py --cv-only
    python scripts/03_train_xgboost.py --gen-test    # test on Scenario 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from c2gnn.models.xgboost_baseline import (  # noqa: E402
    XGBoostC2Detector,
    engineer_features,
    prepare_xy,
)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).parent.parent / "models" / "artifacts"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def load_split(train_path: Path, test_path: Path) -> tuple:
    print(f"  Loading train: {train_path.name}")
    df_train = pl.read_parquet(train_path)
    print(f"  Loading test : {test_path.name}")
    df_test = pl.read_parquet(test_path)

    # CTU-13 has no 'normal' label — background IS the benign class (y=0)
    X_train, y_train = prepare_xy(df_train, exclude_background=False)
    X_test, y_test = prepare_xy(df_test, exclude_background=False)

    print(f"  Train: {X_train.shape}, botnet={y_train.mean():.3f}")
    print(f"  Test : {X_test.shape}, botnet={y_test.mean():.3f}")
    return X_train, y_train, X_test, y_test


def measure_inference_latency(
    detector: XGBoostC2Detector,
    X: np.ndarray,
    n_warmup: int = 100,
    n_measure: int = 500,
) -> dict[str, float]:
    """Measure per-sample inference latency in ms."""
    if detector.model is None:
        return {}

    # Warm up
    _ = detector.model.predict_proba(X[:n_warmup])

    # Measure batch of n_measure individual predictions
    latencies = []
    for i in range(min(n_measure, len(X))):
        sample = X[i : i + 1]
        t0 = time.perf_counter()
        _ = detector.model.predict_proba(sample)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    arr = np.array(latencies)
    return {
        "latency_mean_ms": float(arr.mean()),
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_p99_ms": float(np.percentile(arr, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost C2 detector")
    parser.add_argument("--cv-only", action="store_true", help="Run CV only, no final train")
    parser.add_argument("--gen-test", action="store_true", help="Also test on Scenario 8")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / "scenario10_train.parquet"
    test_path = PROCESSED_DIR / "scenario10_test.parquet"

    if not train_path.exists() or not test_path.exists():
        print("FAIL Preprocessed data not found.")
        print("  Run: python scripts/02_preprocess.py")
        sys.exit(1)

    print("=" * 60)
    print("XGBoost C2 Detector Training")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_split(train_path, test_path)

    detector = XGBoostC2Detector(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        mlflow_experiment="c2-detection-xgboost",
    )

    # -- 5-fold Cross-Validation -----------------------------------------------
    print("\n[1/3] Cross-Validation (5-fold)...")
    cv_metrics = detector.cross_validate(
        X_train, y_train, n_splits=args.n_folds, run_name="xgboost-cv-5fold"
    )
    print(f"  CV F1  : {cv_metrics.get('cv_f1_mean', 0):.4f} ± {cv_metrics.get('cv_f1_std', 0):.4f}")
    print(f"  CV AUC : {cv_metrics.get('cv_auc_mean', 0):.4f} ± {cv_metrics.get('cv_auc_std', 0):.4f}")

    if args.cv_only:
        print("\nCV-only mode — done.")
        return

    # -- Final Training on full train set -------------------------------------
    print("\n[2/3] Final training on full train set...")
    train_metrics = detector.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        run_name="xgboost-final",
    )

    # -- Holistic test set evaluation -----------------------------------------
    print("\n[3/3] Evaluating on held-out test set...")
    test_metrics = detector.evaluate(X_test, y_test)

    # Measure latency
    latency = measure_inference_latency(detector, X_test)

    # False Positive Rate
    from sklearn.metrics import confusion_matrix
    y_pred = detector.model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / max(tn + fp, 1)
    else:
        fpr = 0.0

    all_metrics = {
        **test_metrics,
        **latency,
        "false_positive_rate": float(fpr),
        **cv_metrics,
    }

    print(f"\n  F1       : {all_metrics.get('f1', 0):.4f}")
    print(f"  AUC      : {all_metrics.get('roc_auc', 0):.4f}")
    print(f"  FPR      : {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"  Latency  : {latency.get('latency_mean_ms', 0):.2f} ms/sample")

    # -- Generalization test (Scenario 8) --------------------------------------
    if args.gen_test:
        sc08_path = PROCESSED_DIR / "scenario08_test.parquet"
        if sc08_path.exists():
            print("\n[Gen] Testing on Scenario 8 (Rbot, unseen botnet)...")
            df_gen = pl.read_parquet(sc08_path)
            X_gen, y_gen = prepare_xy(df_gen)
            gen_metrics = detector.evaluate(X_gen, y_gen)
            print(f"  Gen F1  : {gen_metrics.get('f1', 0):.4f}")
            print(f"  Gen AUC : {gen_metrics.get('roc_auc', 0):.4f}")
            all_metrics["gen_f1"] = gen_metrics.get("f1", 0)
            all_metrics["gen_auc"] = gen_metrics.get("roc_auc", 0)
        else:
            print("  Scenario 8 not found — skip generalization test")

    # -- Save model and metrics ------------------------------------------------
    model_path = ARTIFACTS_DIR / "xgboost_model.json"
    detector.model.save_model(str(model_path))
    print(f"\n  Saved model: {model_path.name}")

    metrics_path = ARTIFACTS_DIR / "xgboost_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: round(float(v), 6) for k, v in all_metrics.items()}, f, indent=2)
    print(f"  Saved metrics: {metrics_path.name}")

    # -- SHAP feature importance -----------------------------------------------
    try:
        print("\n  Computing SHAP feature importance...")
        shap_vals = detector.explain(X_test[:1000])
        # shap_vals may be shap.Explanation (SHAP>=0.40) with shape (n, features, 2)
        # or plain ndarray (older SHAP) — handle both
        vals = shap_vals.values if hasattr(shap_vals, "values") else np.array(shap_vals)
        if vals.ndim == 3:
            vals = vals[:, :, 1]  # botnet class only
        mean_abs = np.abs(vals).mean(axis=0)
        feature_names = detector._feature_names
        importance = sorted(
            zip(feature_names, mean_abs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        shap_path = ARTIFACTS_DIR / "shap_feature_importance.json"
        with open(shap_path, "w") as f:
            json.dump({"features": [{"name": n, "shap_mean_abs": round(v, 6)} for n, v in importance]}, f, indent=2)
        print(f"  Saved SHAP: {shap_path.name}")
        print("\n  Top-5 features by SHAP:")
        for name, val in importance[:5]:
            print(f"    {name:<25}: {val:.4f}")
    except Exception as e:
        print(f"  SHAP skipped: {e}")

    # -- Classification report -------------------------------------------------
    from sklearn.metrics import classification_report
    report = classification_report(y_test, detector.model.predict(X_test), target_names=["normal", "botnet"])
    report_path = REPORTS_DIR / "xgboost_classification_report.txt"
    with open(report_path, "w") as f:
        f.write("XGBoost C2 Detector — Classification Report\n")
        f.write("Dataset: CTU-13 Scenario 10\n\n")
        f.write(report)
    print(f"  Saved report: {report_path.name}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"\n  FINAL METRICS (for report):")
    print(f"  ------------------------------------")
    print(f"  - F1 Score    : {all_metrics.get('f1', 0):.4f}              -")
    print(f"  - AUC-ROC     : {all_metrics.get('roc_auc', 0):.4f}              -")
    print(f"  - FPR         : {fpr*100:.2f}%                -")
    print(f"  - Latency     : {latency.get('latency_mean_ms', 0):.2f} ms/sample        -")
    print(f"  ------------------------------------")
    print("\nNext step (optional — requires torch):")
    print("  python scripts/04_train_gnn.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
