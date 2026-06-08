"""
xgboost_baseline.py — XGBoost Tabular Baseline for C2 Detection

So sánh với GNN để chứng minh lợi ích của graph structure.
Baseline quan trọng — phải mạnh để GNN improvement có ý nghĩa.

Author: Member 2 (AI/GNN/MLOps Engineer)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import structlog
import polars as pl

logger = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

TABULAR_FEATURES = [
    "duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_bytes",
    "packet_rate",
    "byte_rate",
    "flow_iat_mean",
    "flow_iat_std",
    "src_port",
    "dst_port",
    # Derived
    "bytes_per_packet",
    "fwd_bwd_ratio",
    "dst_port_well_known",
    "is_tcp",
    "is_udp",
    "is_icmp",
]


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Feature engineering on top of raw flow data.
    Using Polars for speed (5x faster than pandas for this).
    """
    return df.with_columns([
        (pl.col("total_bytes") / (pl.col("total_fwd_packets") + pl.col("total_bwd_packets") + 1))
        .alias("bytes_per_packet"),

        (pl.col("total_fwd_packets") / (pl.col("total_bwd_packets") + 1))
        .alias("fwd_bwd_ratio"),

        ((pl.col("dst_port") > 0) & (pl.col("dst_port") < 1024))
        .cast(pl.Float32).alias("dst_port_well_known"),

        (pl.col("protocol") == "TCP").cast(pl.Float32).alias("is_tcp"),
        (pl.col("protocol") == "UDP").cast(pl.Float32).alias("is_udp"),
        (pl.col("protocol") == "ICMP").cast(pl.Float32).alias("is_icmp"),
    ])


def prepare_xy(
    df: pl.DataFrame,
    feature_cols: Optional[list[str]] = None,
    binary: bool = True,
    exclude_background: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare X, y numpy arrays for sklearn/XGBoost.

    Args:
        exclude_background: If True, trains only on normal vs botnet.
                           Better for precision/recall; may overfit if background differs.
    """
    if exclude_background:
        df = df.filter(pl.col("label") != "background")

    features = feature_cols or TABULAR_FEATURES
    df = engineer_features(df)

    # Ensure all feature columns exist
    missing = set(features) - set(df.columns)
    if missing:
        logger.warning("Missing feature columns, filling with 0", missing=list(missing))
        for col in missing:
            df = df.with_columns(pl.lit(0.0).alias(col))

    X = df.select(features).to_numpy().astype(np.float32)
    y = (df["label"] == "botnet").to_numpy().astype(np.int32)

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost Detector
# ─────────────────────────────────────────────────────────────────────────────


class XGBoostC2Detector:
    """
    XGBoost C2 flow classifier.

    Key design choices:
    - Automatic class weight adjustment for imbalanced CTU-13
    - 5-fold CV for robust evaluation
    - SHAP TreeExplainer for feature importance
    - MLflow logging for experiment tracking
    """

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 7,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        mlflow_experiment: str = "c2-detection-xgboost",
        random_state: int = 42,
    ):
        self._base_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "random_state": random_state,
            "tree_method": "hist",   # Fast CPU training
            "device": "cuda" if self._has_gpu() else "cpu",
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        self.experiment = mlflow_experiment
        self.model: Optional[xgb.XGBClassifier] = None
        self._explainer: Optional[shap.TreeExplainer] = None
        self._feature_names: list[str] = TABULAR_FEATURES

    @staticmethod
    def _has_gpu() -> bool:
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
        run_name: str = "xgboost-baseline",
    ) -> dict[str, float]:
        """
        Train with MLflow tracking.
        Returns: metrics dict
        """
        if feature_names:
            self._feature_names = feature_names

        # Auto scale_pos_weight for class imbalance
        botnet_rate = y_train.mean()
        scale_pos_weight = (1.0 - botnet_rate) / max(botnet_rate, 1e-6)
        logger.info(
            "Class balance",
            botnet_rate=round(botnet_rate, 4),
            scale_pos_weight=round(scale_pos_weight, 2),
        )

        params = {**self._base_params, "scale_pos_weight": scale_pos_weight}

        mlflow.set_experiment(self.experiment)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_params({
                "train_samples": len(X_train),
                "botnet_rate": round(botnet_rate, 4),
                "n_features": X_train.shape[1],
                "exclude_background": True,
            })

            self.model = xgb.XGBClassifier(**params)

            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))

            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=50,
            )

            # Evaluate on validation (or training if no val)
            X_eval = X_val if X_val is not None else X_train
            y_eval = y_val if y_val is not None else y_train
            metrics = self.evaluate(X_eval, y_eval)

            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(
                self.model,
                artifact_path="model",
                input_example=X_train[:5],
            )

            logger.info(
                "Training complete",
                run_id=run.info.run_id,
                **{k: round(v, 4) for k, v in metrics.items()},
            )

        return metrics

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        run_name: str = "xgboost-cv",
    ) -> dict[str, float]:
        """
        Stratified K-Fold CV for robust evaluation.
        Reports mean ± std across folds.
        """
        botnet_rate = y.mean()
        scale_pos_weight = (1.0 - botnet_rate) / max(botnet_rate, 1e-6)
        params = {**self._base_params, "scale_pos_weight": scale_pos_weight}

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        all_f1, all_auc, all_pr_auc = [], [], []

        mlflow.set_experiment(self.experiment)

        with mlflow.start_run(run_name=run_name):
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1]

                fold_f1 = f1_score(y_val, y_pred, pos_label=1)
                fold_auc = roc_auc_score(y_val, y_prob)
                fold_pr = average_precision_score(y_val, y_prob)

                all_f1.append(fold_f1)
                all_auc.append(fold_auc)
                all_pr_auc.append(fold_pr)

                mlflow.log_metrics(
                    {f"fold{fold}_f1": fold_f1, f"fold{fold}_auc": fold_auc},
                )
                logger.info(
                    f"Fold {fold}",
                    f1=round(fold_f1, 4),
                    auc=round(fold_auc, 4),
                    pr_auc=round(fold_pr, 4),
                )

            cv_metrics = {
                "cv_f1_mean": float(np.mean(all_f1)),
                "cv_f1_std": float(np.std(all_f1)),
                "cv_auc_mean": float(np.mean(all_auc)),
                "cv_auc_std": float(np.std(all_auc)),
                "cv_pr_auc_mean": float(np.mean(all_pr_auc)),
                "cv_pr_auc_std": float(np.std(all_pr_auc)),
            }
            mlflow.log_metrics(cv_metrics)

        return cv_metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Full evaluation suite."""
        assert self.model is not None, "Model not trained"

        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]

        print("\n" + "="*60)
        print(classification_report(y, y_pred, target_names=["normal/bg", "botnet"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))
        print("="*60)

        return {
            "f1_botnet": f1_score(y, y_pred, pos_label=1, zero_division=0),
            "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(y, y_prob),
            "pr_auc": average_precision_score(y, y_prob),
            "precision_botnet": float(
                (y_pred[y == 1] == 1).sum() / max((y_pred == 1).sum(), 1)
            ),
            "recall_botnet": float(
                (y_pred[y == 1] == 1).sum() / max((y == 1).sum(), 1)
            ),
        }

    def explain(
        self,
        X: np.ndarray,
        max_samples: int = 500,
        save_path: Optional[Path] = None,
    ) -> shap.Explanation:
        """
        SHAP TreeExplainer for global + local feature importance.
        Important: shows WHICH features signal C2 behavior.
        """
        assert self.model is not None
        import matplotlib.pyplot as plt

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)

        sample = X[:max_samples] if len(X) > max_samples else X
        shap_values = self._explainer(sample, feature_names=self._feature_names)

        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values[:, :, 1],  # SHAP for botnet class
            sample,
            feature_names=self._feature_names,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("SHAP plot saved", path=str(save_path))
        else:
            plt.show()

        plt.close()
        return shap_values

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return botnet probability for each flow."""
        assert self.model is not None
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        assert self.model is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info("Model saved", path=str(path))

    @classmethod
    def load(cls, path: Path, **kwargs) -> "XGBoostC2Detector":
        detector = cls(**kwargs)
        detector.model = xgb.XGBClassifier()
        detector.model.load_model(str(path))
        return detector


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark inference speed
# ─────────────────────────────────────────────────────────────────────────────


def benchmark_inference_latency(
    model: XGBoostC2Detector,
    X: np.ndarray,
    n_iterations: int = 100,
) -> dict[str, float]:
    """
    Measure inference latency per batch.
    Compare with GNN inference for the latency analysis section.
    """
    import time

    latencies = []
    batch_sizes = [1, 10, 100, 1000]
    results = {}

    for bs in batch_sizes:
        batch = X[:bs]
        times = []

        for _ in range(n_iterations):
            t0 = time.perf_counter()
            _ = model.predict_proba(batch)
            times.append((time.perf_counter() - t0) * 1000)

        results[f"bs{bs}_p50_ms"] = float(np.percentile(times, 50))
        results[f"bs{bs}_p95_ms"] = float(np.percentile(times, 95))
        logger.info(
            "Latency benchmark",
            batch_size=bs,
            p50_ms=round(results[f"bs{bs}_p50_ms"], 2),
            p95_ms=round(results[f"bs{bs}_p95_ms"], 2),
        )

    return results
