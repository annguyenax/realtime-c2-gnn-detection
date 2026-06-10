"""
graphsage.py — GraphSAGE + GATv2 for Node-Level C2 Detection

Key design choices:
  - GraphSAGE: inductive learning (handles unseen IPs at inference time)
  - GATv2: attention mechanism for neighbor importance weighting + explainability
  - Both: BatchNorm after each layer, Dropout for regularization
  - Training: class-weighted loss for imbalanced botnet labels

Author: Member 2 (AI/GNN/MLOps Engineer)
"""

from __future__ import annotations

import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, GATv2Conv, SAGEConv

logger = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants — MUST match NodeData.to_feature_vector() and EdgeData
# ─────────────────────────────────────────────────────────────────────────────
NODE_FEATURE_DIM = 18  # 14 flow stats + 4 temporal (active_span, mean_iat, iat_cv, repeat_dst_ratio)
EDGE_FEATURE_DIM = 7   # EdgeData.to_feature_vector() minus botnet_fraction (last dim)


# ─────────────────────────────────────────────────────────────────────────────
# GraphSAGE
# ─────────────────────────────────────────────────────────────────────────────


class GraphSAGEC2Detector(nn.Module):
    """
    GraphSAGE-based node classifier for C2 botnet detection.

    Why GraphSAGE over vanilla GCN?
      - Inductive: works on unseen nodes (new IPs not in training)
      - Scalable: mini-batch training via neighbor sampling
      - Aggregation: mean/max/LSTM of neighbor features

    Architecture:
      Input (14) → SAGE(64) → BN → ReLU → Dropout
                → SAGE(64) → BN → ReLU → Dropout
                → SAGE(64) → BN → ReLU
                → MLP(64→32→2)
    """

    def __init__(
        self,
        in_channels: int = NODE_FEATURE_DIM,
        hidden_channels: int = 128,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
        aggr: str = "mean",  # "mean" | "max" | "lstm"
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        dims = [in_channels] + [hidden_channels] * num_layers

        self.convs = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1], aggr=aggr) for i in range(num_layers)]
        )

        self.bns = nn.ModuleList([BatchNorm(dims[i + 1]) for i in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,  # Unused in SAGE, kept for API consistency
    ) -> torch.Tensor:
        """
        Args:
            x:          [num_nodes, in_channels]
            edge_index: [2, num_edges]
            edge_attr:  [num_edges, edge_feat] (ignored by SAGE, used by GAT)
        Returns:
            logits:     [num_nodes, out_channels]
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Extract node embeddings (pre-classifier) for visualization."""
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# GATv2
# ─────────────────────────────────────────────────────────────────────────────


class GATv2C2Detector(nn.Module):
    """
    GATv2-based node classifier for C2 detection.

    Why GATv2 over original GAT?
      - GATv2 (Brody et al. 2022) fixes static attention problem in original GAT
      - More expressive: dynamic attention adapts to query node
      - Edge features: can incorporate flow statistics into attention computation
      - Attention weights: visualize which neighbors influence the detection

    Architecture:
      Input (14) → GATv2(64, heads=4, concat) → BN → ELU → Dropout
                → GATv2(64, heads=4, concat) → BN → ELU → Dropout
                → GATv2(64, heads=1) → BN → ELU
                → MLP(64→32→2)
    """

    def __init__(
        self,
        in_channels: int = NODE_FEATURE_DIM,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        edge_dim: int = EDGE_FEATURE_DIM,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            is_last_layer = i == num_layers - 1

            in_ch = in_channels if i == 0 else hidden_channels * heads
            n_heads = 1 if is_last_layer else heads
            concat = not is_last_layer

            self.convs.append(
                GATv2Conv(
                    in_channels=in_ch,
                    out_channels=hidden_channels,
                    heads=n_heads,
                    concat=concat,
                    dropout=dropout,
                    edge_dim=edge_dim if i == 0 else None,  # Only 1st layer uses edge attr
                    add_self_loops=False,
                )
            )

            out_size = hidden_channels * heads if concat else hidden_channels
            self.bns.append(BatchNorm(out_size))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

        # Store attention weights for explainability
        self._last_attention: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:                [num_nodes, in_channels]
            edge_index:       [2, num_edges]
            edge_attr:        [num_edges, edge_feat] — used in layer 0
            return_attention: If True, stores attention in self._last_attention
        Returns:
            logits: [num_nodes, out_channels]
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            ea = edge_attr if i == 0 else None

            if return_attention and i == 0:
                x, attn = conv(x, edge_index, edge_attr=ea, return_attention_weights=True)
                self._last_attention = attn  # (edge_index, alpha): useful for visualization
            else:
                x = conv(x, edge_index, edge_attr=ea)

            x = bn(x)
            x = F.elu(x)  # ELU works better with attention-based models
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)

    def get_attention_for_node(self, node_idx: int) -> dict | None:
        """
        Extract attention weights for neighbors of a specific node.
        Used for explainability: "why is this IP suspicious?"

        Returns: {neighbor_ip_idx: attention_weight}
        """
        if self._last_attention is None:
            return None

        attn_edge_index, attn_weights = self._last_attention
        # attn_weights: [num_edges, num_heads] — take mean across heads

        mask = attn_edge_index[1] == node_idx  # Edges pointing TO this node
        if not mask.any():
            return {}

        src_nodes = attn_edge_index[0][mask].cpu().numpy()
        weights = attn_weights[mask].mean(dim=1).cpu().detach().numpy()

        return {int(src): float(w) for src, w in zip(src_nodes, weights)}


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., RetinaNet 2017) for imbalanced node classification.

    Down-weights easy-to-classify examples so training focuses on hard examples
    near the decision boundary — addressing the "easy background dominates" problem.

    WARNING — double reweighting:
      Do NOT combine high alpha class weights (e.g. [1, 50]) with high gamma (>1.5).
      The two mechanisms compound: the model gets pushed too aggressively toward the
      positive class, precision collapses even worse than vanilla weighted CE.

    Recommended experiments (in this order):
      1. FocalLoss(alpha=None, gamma=1.5)      — focal only, baseline
      2. FocalLoss(alpha=None, gamma=2.0)      — stronger focus on hard examples
      3. FocalLoss(alpha=[1.0, 5.0], gamma=1.5) — light class balance + focal

    gamma=0 is equivalent to standard CrossEntropyLoss (with optional alpha).
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 1.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            target,
            weight=self.alpha,
            reduction="none",
        )
        # p_t = probability of the correct class; high p_t = easy example
        pt = torch.exp(-ce_loss)
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss

        if self.reduction == "sum":
            return focal_loss.sum()
        if self.reduction == "none":
            return focal_loss
        return focal_loss.mean()


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.40,
) -> dict[str, float]:
    """
    Find decision threshold that maximises F1 subject to a minimum recall constraint.

    CRITICAL — call this on VALIDATION data only, never on test data.
    Save the returned threshold and apply it consistently to test evaluation
    and real-time inference. Tuning on test data is information leakage.

    Args:
        y_true:     Ground truth labels (0/1), node-level
        y_prob:     Model output probabilities for class 1 (botnet)
        min_recall: Hard floor on recall — thresholds that drop recall below
                    this are skipped. 0.40 keeps at least 40% botnet detection.
                    Tune based on your SOC tolerance for missed detections.

    Returns:
        dict with keys: threshold, f1, precision, recall
    """
    from sklearn.metrics import precision_recall_curve

    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)

    best: dict[str, float] = {
        "threshold": 0.5,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    for p, r, t in zip(precision_arr[:-1], recall_arr[:-1], thresholds):
        if r < min_recall:
            continue
        f1 = 2.0 * p * r / max(p + r, 1e-12)
        if f1 > best["f1"]:
            best = {
                "threshold": float(t),
                "f1": float(f1),
                "precision": float(p),
                "recall": float(r),
            }

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────


class GNNTrainer:
    """
    Training loop for GraphSAGE / GATv2.

    Features:
    - Weighted CrossEntropyLoss for class imbalance
    - Early stopping on validation F1
    - MLflow experiment tracking
    - Checkpoint saving (best model only)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        mlflow_experiment: str = "c2-detection-gnn",
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.experiment = mlflow_experiment

    def train(
        self,
        train_graphs: list[Data],
        val_graphs: list[Data],
        epochs: int = 100,
        patience: int = 15,
        save_path: Path | None = None,
        run_name: str = "gnn-train",
        max_class_weight: float = 50.0,
        filter_empty_snapshots: bool = True,
        use_focal_loss: bool = False,
        focal_gamma: float = 1.5,
        tune_threshold: bool = True,
        min_recall: float = 0.40,
    ) -> dict[str, float]:
        """
        Train on list of PyG Data objects (one per time window).

        Args:
            train_graphs:          List of PyG Data (each is a graph snapshot)
            val_graphs:            Validation graphs (time-split, not random!)
            patience:              Early stopping patience (epochs without val improvement)
            max_class_weight:      Cap for botnet class weight (prevents precision collapse
                                   when raw n_neg/n_pos exceeds 1000)
            filter_empty_snapshots: Remove all-normal snapshots from training
            use_focal_loss:        Replace weighted CE with FocalLoss(gamma=focal_gamma).
                                   When True, class weight is NOT applied (avoid double
                                   reweighting — see FocalLoss docstring).
            focal_gamma:           Focal loss concentration parameter. Start at 1.5;
                                   increase to 2.0 only if precision is still low.
            tune_threshold:        After training, find optimal threshold on val_graphs
                                   and include it in the returned metrics dict.
            min_recall:            Minimum recall floor for threshold search.
        """
        model_name = self.model.__class__.__name__
        mlflow.set_experiment(self.experiment)

        # Filter all-normal snapshots before computing class weights
        if filter_empty_snapshots:
            n_before = len(train_graphs)
            train_graphs = [
                g for g in train_graphs
                if hasattr(g, "y") and g.y is not None and g.y.sum().item() > 0
            ]
            logger.info(
                "Filtered empty snapshots",
                before=n_before,
                after=len(train_graphs),
                kept_pct=round(len(train_graphs) / max(n_before, 1) * 100, 1),
            )

        with mlflow.start_run(run_name=f"{run_name}-{model_name}") as run:
            # Compute class weight; FocalLoss handles imbalance internally via
            # gamma so we skip alpha to avoid double reweighting.
            all_y = torch.cat([g.y for g in train_graphs if hasattr(g, "y")])
            n_pos = all_y.sum().item()
            n_neg = len(all_y) - n_pos
            raw_weight = n_neg / max(n_pos, 1)
            capped_weight = min(raw_weight, max_class_weight)

            if use_focal_loss:
                criterion = FocalLoss(alpha=None, gamma=focal_gamma)
            else:
                weight = torch.tensor(
                    [1.0, capped_weight],
                    dtype=torch.float32,
                    device=self.device,
                )
                criterion = nn.CrossEntropyLoss(weight=weight)

            # Log hyperparams
            mlflow.log_params(
                {
                    "model": model_name,
                    "epochs": epochs,
                    "patience": patience,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
                    "device": str(self.device),
                    "train_graphs": len(train_graphs),
                    "val_graphs": len(val_graphs),
                    "loss_fn": "FocalLoss" if use_focal_loss else "WeightedCE",
                    "focal_gamma": focal_gamma if use_focal_loss else 0.0,
                    "max_class_weight": max_class_weight,
                    "filter_empty_snapshots": filter_empty_snapshots,
                    "raw_class_weight_botnet": round(raw_weight, 2),
                    "capped_class_weight_botnet": round(capped_weight, 2) if not use_focal_loss else "n/a",
                }
            )

            logger.info(
                "Training start",
                model=model_name,
                n_train=len(train_graphs),
                n_val=len(val_graphs),
                raw_class_weight_botnet=round(raw_weight, 2),
                capped_class_weight_botnet=round(capped_weight, 2),
                device=str(self.device),
            )

            best_val_f1 = 0.0
            best_epoch = 0
            no_improve = 0

            for epoch in range(1, epochs + 1):
                # ── Train ────────────────────────────────────────────────────
                train_loss = self._train_epoch(train_graphs, criterion)
                self.scheduler.step()

                # ── Validate ─────────────────────────────────────────────────
                val_metrics = self._eval_epoch(val_graphs)
                val_f1 = val_metrics["f1_botnet"]

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                    },
                    step=epoch,
                )

                # ── Early stopping ───────────────────────────────────────────
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch = epoch
                    no_improve = 0

                    if save_path:
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(self.model.state_dict(), save_path)
                        logger.info(
                            f"Checkpoint saved (epoch {epoch})",
                            val_f1=round(val_f1, 4),
                        )
                else:
                    no_improve += 1

                if epoch % 10 == 0 or no_improve == patience:
                    logger.info(
                        f"Epoch {epoch:3d}/{epochs}",
                        loss=round(train_loss, 4),
                        val_f1=round(val_f1, 4),
                        val_auc=round(val_metrics["roc_auc"], 4),
                        best_epoch=best_epoch,
                        no_improve=no_improve,
                    )

                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Load best checkpoint for threshold tuning and final metrics
            if save_path and save_path.exists():
                self.model.load_state_dict(torch.load(save_path, map_location=self.device))
                mlflow.pytorch.log_model(self.model, "best_model")

            # Threshold tuning on val_graphs — NOT on test data (would be leakage)
            optimal_threshold = 0.5
            if tune_threshold:
                val_probs, val_labels = self._collect_probs(val_graphs)
                if len(np.unique(val_labels)) > 1:
                    thr_info = find_best_threshold(val_labels, val_probs, min_recall=min_recall)
                    optimal_threshold = thr_info["threshold"]
                    logger.info(
                        "Optimal threshold (from val)",
                        threshold=round(optimal_threshold, 4),
                        val_f1=round(thr_info["f1"], 4),
                        val_precision=round(thr_info["precision"], 4),
                        val_recall=round(thr_info["recall"], 4),
                    )
                    mlflow.log_metrics({
                        "optimal_threshold": optimal_threshold,
                        "val_f1_at_optimal_thr": thr_info["f1"],
                        "val_precision_at_optimal_thr": thr_info["precision"],
                        "val_recall_at_optimal_thr": thr_info["recall"],
                    })

            # Final metrics
            final_metrics = {
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch,
                "optimal_threshold": optimal_threshold,
            }
            mlflow.log_metrics({"best_val_f1": best_val_f1, "best_epoch": best_epoch})

            logger.info(
                "Training complete",
                best_epoch=best_epoch,
                best_val_f1=round(best_val_f1, 4),
                optimal_threshold=round(optimal_threshold, 4),
                run_id=run.info.run_id,
            )

        return final_metrics

    def _train_epoch(self, graphs: list[Data], criterion: nn.Module) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # Shuffle graphs each epoch
        indices = torch.randperm(len(graphs))

        for idx in indices:
            data = graphs[idx].to(self.device)

            if not hasattr(data, "y") or data.y is None:
                continue

            self.optimizer.zero_grad()

            edge_attr = data.edge_attr
            if edge_attr is not None and edge_attr.shape[1] > EDGE_FEATURE_DIM:
                edge_attr = edge_attr[:, :EDGE_FEATURE_DIM]  # Strip ground truth dim

            logits = self.model(data.x, data.edge_index, edge_attr)
            loss = criterion(logits, data.y)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval_epoch(self, graphs: list[Data]) -> dict[str, float]:
        self.model.eval()

        all_probs, all_labels = [], []

        for data in graphs:
            data = data.to(self.device)
            if not hasattr(data, "y") or data.y is None:
                continue

            edge_attr = data.edge_attr
            if edge_attr is not None and edge_attr.shape[1] > EDGE_FEATURE_DIM:
                edge_attr = edge_attr[:, :EDGE_FEATURE_DIM]

            logits = self.model(data.x, data.edge_index, edge_attr)
            probs = torch.softmax(logits, dim=-1)[:, 1]

            all_probs.append(probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

        if not all_probs:
            return {"f1_botnet": 0.0, "roc_auc": 0.0, "pr_auc": 0.0}

        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels)
        y_pred = (y_prob >= 0.5).astype(int)

        return {
            "f1_botnet": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
            "pr_auc": average_precision_score(y_true, y_prob)
            if len(np.unique(y_true)) > 1
            else 0.0,
        }

    @torch.no_grad()
    def _collect_probs(
        self, graphs: list[Data]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect (y_prob, y_true) across all graphs. Used for threshold tuning."""
        self.model.eval()
        all_probs, all_labels = [], []
        for data in graphs:
            data = data.to(self.device)
            if not hasattr(data, "y") or data.y is None:
                continue
            ea = data.edge_attr
            if ea is not None and ea.shape[1] > EDGE_FEATURE_DIM:
                ea = ea[:, :EDGE_FEATURE_DIM]
            logits = self.model(data.x, data.edge_index, ea)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
        if not all_probs:
            return np.array([]), np.array([])
        return np.concatenate(all_probs), np.concatenate(all_labels)

    @torch.no_grad()
    def benchmark_inference(self, test_graphs: list[Data], n_warmup: int = 5) -> dict[str, float]:
        """
        Measure per-graph inference latency.
        Critical for latency analysis section of report.
        """
        self.model.eval()
        latencies_ms = []

        # Warmup
        for data in test_graphs[:n_warmup]:
            data = data.to(self.device)
            ea = data.edge_attr
            if ea is not None and ea.shape[1] > EDGE_FEATURE_DIM:
                ea = ea[:, :EDGE_FEATURE_DIM]
            _ = self.model(data.x, data.edge_index, ea)

        for data in test_graphs:
            data = data.to(self.device)
            edge_attr = data.edge_attr
            if edge_attr is not None and edge_attr.shape[1] > EDGE_FEATURE_DIM:
                edge_attr = edge_attr[:, :EDGE_FEATURE_DIM]
            t0 = time.perf_counter()
            _ = self.model(data.x, data.edge_index, edge_attr)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

        return {
            "inference_p50_ms": float(np.percentile(latencies_ms, 50)),
            "inference_p95_ms": float(np.percentile(latencies_ms, 95)),
            "inference_p99_ms": float(np.percentile(latencies_ms, 99)),
            "inference_mean_ms": float(np.mean(latencies_ms)),
        }
