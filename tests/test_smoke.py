"""Smoke tests using synthetic data — no real data download needed.

Verifies that:
1. Models can be instantiated
2. Forward passes produce correct output shapes
3. Loss computes and is finite
4. One training step completes without error
5. Metrics compute correctly
"""

from __future__ import annotations

import torch
import pytest
from torch_geometric.data import Data

from trade_flow_gcn.models.gcn import TradeFlowGCN
from trade_flow_gcn.models.mlp_baseline import MLPBaseline
from trade_flow_gcn.evaluation.metrics import compute_all_metrics, rmse, mae, r_squared


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_graph() -> Data:
    """Create a small synthetic trade graph for testing.

    10 nodes (countries), ~30 directed edges.
    """
    num_nodes = 10
    node_feat_dim = 3
    edge_feat_dim = 6
    num_edges = 30

    x = torch.randn(num_nodes, node_feat_dim)
    # Random directed edges (no self-loops)
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    # Remove self-loops
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index = torch.stack([src, dst])

    actual_edges = edge_index.size(1)
    edge_attr = torch.randn(actual_edges, edge_feat_dim)
    y = torch.randn(actual_edges).abs()  # log-trade is non-negative

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.year = 2010
    return data


# ── Model Tests ───────────────────────────────────────────────────────


class TestTradeFlowGCN:
    """Tests for the GCN model."""

    def test_instantiation(self):
        model = TradeFlowGCN(node_input_dim=3, edge_input_dim=6)
        assert model is not None

    def test_forward_shape(self, synthetic_graph: Data):
        model = TradeFlowGCN(node_input_dim=3, edge_input_dim=6)
        model.eval()
        with torch.no_grad():
            pred = model(
                synthetic_graph.x,
                synthetic_graph.edge_index,
                synthetic_graph.edge_attr,
            )
        assert pred.shape == synthetic_graph.y.shape

    def test_forward_finite(self, synthetic_graph: Data):
        model = TradeFlowGCN(node_input_dim=3, edge_input_dim=6)
        model.eval()
        with torch.no_grad():
            pred = model(
                synthetic_graph.x,
                synthetic_graph.edge_index,
                synthetic_graph.edge_attr,
            )
        assert torch.isfinite(pred).all()

    def test_training_step(self, synthetic_graph: Data):
        model = TradeFlowGCN(node_input_dim=3, edge_input_dim=6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()

        pred = model(
            synthetic_graph.x,
            synthetic_graph.edge_index,
            synthetic_graph.edge_attr,
        )
        loss = torch.nn.functional.mse_loss(pred, synthetic_graph.y)

        assert torch.isfinite(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class TestMLPBaseline:
    """Tests for the MLP baseline model."""

    def test_instantiation(self):
        model = MLPBaseline(input_dim=12)
        assert model is not None

    def test_forward_shape(self, synthetic_graph: Data):
        model = MLPBaseline(input_dim=12)
        model.eval()
        with torch.no_grad():
            pred = model(
                synthetic_graph.x,
                synthetic_graph.edge_index,
                synthetic_graph.edge_attr,
            )
        assert pred.shape == synthetic_graph.y.shape

    def test_training_step(self, synthetic_graph: Data):
        model = MLPBaseline(input_dim=12)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()

        pred = model(
            synthetic_graph.x,
            synthetic_graph.edge_index,
            synthetic_graph.edge_attr,
        )
        loss = torch.nn.functional.mse_loss(pred, synthetic_graph.y)

        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ── Metric Tests ──────────────────────────────────────────────────────


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_rmse_perfect(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        assert rmse(y, y).item() == pytest.approx(0.0, abs=1e-6)

    def test_mae_perfect(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        assert mae(y, y).item() == pytest.approx(0.0, abs=1e-6)

    def test_r2_perfect(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        assert r_squared(y, y).item() == pytest.approx(1.0, abs=1e-6)

    def test_compute_all_returns_dict(self):
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([1.1, 2.2, 2.8])
        metrics = compute_all_metrics(y_pred, y_true)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mape" in metrics

    def test_rmse_value(self):
        y_pred = torch.tensor([1.0, 2.0])
        y_true = torch.tensor([1.0, 4.0])
        # RMSE = sqrt(mean([0, 4])) = sqrt(2) ≈ 1.414
        assert rmse(y_pred, y_true).item() == pytest.approx(2**0.5, abs=1e-4)


# ── Lightning Module Test ─────────────────────────────────────────────


class TestLightningModule:
    """Tests for the Lightning training module."""

    def test_forward(self, synthetic_graph: Data):
        from trade_flow_gcn.training.lightning_module import TradeFlowModule

        model = TradeFlowGCN(node_input_dim=3, edge_input_dim=6)
        lit = TradeFlowModule(model=model)
        lit.eval()
        with torch.no_grad():
            pred = lit(synthetic_graph)
        assert pred.shape == synthetic_graph.y.shape
