"""
Integration test: graph snapshot → GNN forward pass.

Skipped automatically if torch / torch_geometric are not installed.
Verifies end-to-end: FlowRecord → SlidingWindowGraph → PyG Data → model output.
"""

import pytest

pytest.importorskip("torch", reason="requires PyTorch")
pytest.importorskip("torch_geometric", reason="requires PyTorch Geometric")

import torch  # noqa: E402

from c2gnn.graph.dynamic_graph import SlidingWindowGraph  # noqa: E402
from c2gnn.models.graphsage import (  # noqa: E402
    GATv2C2Detector,
    GraphSAGEC2Detector,
)
from tests.conftest import make_flow  # noqa: E402

NODE_IN = 14
EDGE_IN_INFERENCE = 7  # strip botnet_fraction for inference


def _build_snapshot(n_botnet: int = 5, n_normal: int = 20):
    """Return a PyG Data snapshot with mixed botnet/normal flows."""
    g = SlidingWindowGraph(window_size=60.0, edge_ttl=120.0)
    for i in range(n_botnet):
        g.update(make_flow("bot", f"c2_{i}", timestamp=float(1000 + i), label="botnet"))
    for i in range(n_normal):
        src = f"host_{i % 5}"
        dst = f"server_{i % 3}"
        g.update(make_flow(src, dst, timestamp=float(1000 + i), label="normal"))
    return g.to_pyg_data(include_ground_truth=True)


class TestGraphSAGEForwardPass:
    def test_output_shape(self):
        data = _build_snapshot()
        assert data is not None
        model = GraphSAGEC2Detector(in_channels=NODE_IN, hidden_channels=128, out_channels=2)
        model.eval()
        edge_attr = data.edge_attr
        if edge_attr is not None and edge_attr.shape[1] > EDGE_IN_INFERENCE:
            edge_attr = edge_attr[:, :EDGE_IN_INFERENCE]
        with torch.no_grad():
            out = model(data.x, data.edge_index, edge_attr)
        assert out.shape == (data.x.shape[0], 2), f"Expected ({data.x.shape[0]}, 2), got {out.shape}"

    def test_probabilities_sum_to_one(self):
        data = _build_snapshot()
        assert data is not None
        model = GraphSAGEC2Detector(in_channels=NODE_IN, hidden_channels=128, out_channels=2)
        model.eval()
        edge_attr = data.edge_attr
        if edge_attr is not None and edge_attr.shape[1] > EDGE_IN_INFERENCE:
            edge_attr = edge_attr[:, :EDGE_IN_INFERENCE]
        with torch.no_grad():
            out = model(data.x, data.edge_index, edge_attr)
            probs = torch.softmax(out, dim=-1)
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_labels_present(self):
        data = _build_snapshot()
        assert data is not None
        assert data.y is not None
        assert data.y.shape[0] == data.x.shape[0]
        # At least one botnet node should be labeled 1
        assert data.y.sum().item() >= 1


class TestGATv2ForwardPass:
    def test_output_shape(self):
        data = _build_snapshot()
        assert data is not None
        model = GATv2C2Detector(in_channels=NODE_IN, hidden_channels=64, out_channels=2, heads=4)
        model.eval()
        edge_attr = data.edge_attr
        if edge_attr is not None and edge_attr.shape[1] > EDGE_IN_INFERENCE:
            edge_attr = edge_attr[:, :EDGE_IN_INFERENCE]
        with torch.no_grad():
            out = model(data.x, data.edge_index, edge_attr)
        assert out.shape == (data.x.shape[0], 2)

    def test_no_ground_truth_leakage_in_inference_snapshot(self):
        """Inference snapshot (include_ground_truth=False) must have edge_attr dim=7."""
        g = SlidingWindowGraph(window_size=60.0, edge_ttl=120.0)
        for i in range(10):
            g.update(make_flow("src", f"dst_{i}", timestamp=float(1000 + i), label="botnet"))
        data = g.to_pyg_data(include_ground_truth=False)
        assert data is not None
        if data.edge_attr is not None:
            assert data.edge_attr.shape[1] == EDGE_IN_INFERENCE, (
                f"Inference snapshot must not include botnet_fraction (dim 8), "
                f"got shape {data.edge_attr.shape}"
            )

    def test_attention_weights_returned(self):
        data = _build_snapshot()
        assert data is not None
        model = GATv2C2Detector(in_channels=NODE_IN, hidden_channels=64, out_channels=2, heads=4)
        model.eval()
        edge_attr = data.edge_attr
        if edge_attr is not None and edge_attr.shape[1] > EDGE_IN_INFERENCE:
            edge_attr = edge_attr[:, :EDGE_IN_INFERENCE]
        with torch.no_grad():
            _ = model(data.x, data.edge_index, edge_attr, return_attention=True)
        assert model._last_attention is not None
