"""Tests that verify a minimal GCN forward pass produces correct shapes."""

import torch
import torch.nn.functional as F
import pytest

# ---------------------------------------------------------------------------
# Minimal GCN implementation (mirrors scripts/train_gcn.py)
# ---------------------------------------------------------------------------

try:
    from torch_geometric.nn import GCNConv as _GCNConv  # noqa: F401
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


def _make_ring_graph(num_nodes: int) -> torch.Tensor:
    """Return edge_index for a simple ring graph with *num_nodes* nodes."""
    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ])
    return edge_index


# ---------------------------------------------------------------------------
# Pure-PyTorch GCN (no torch_geometric dependency)
# ---------------------------------------------------------------------------

import torch.nn as nn


class SimpleGCNLayer(nn.Module):
    """One GCN layer: A_hat @ X @ W  (without normalization for simplicity)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        # Aggregate neighbour features (sum)
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        return self.linear(agg)


class SimpleGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int) -> None:
        super().__init__()
        self.conv1 = SimpleGCNLayer(in_channels, hidden_channels)
        self.conv2 = SimpleGCNLayer(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_nodes,in_ch,hidden_ch,out_ch", [
    (10, 16, 32, 7),
    (50, 1433, 64, 7),
    (1, 8, 4, 3),
])
def test_gcn_output_shape(num_nodes: int, in_ch: int,
                          hidden_ch: int, out_ch: int) -> None:
    """GCN forward pass must return (num_nodes, out_channels)."""
    model = SimpleGCN(in_ch, hidden_ch, out_ch)
    model.eval()
    x = torch.randn(num_nodes, in_ch)
    edge_index = _make_ring_graph(num_nodes)
    out = model(x, edge_index)
    assert out.shape == (num_nodes, out_ch), (
        f"Expected ({num_nodes}, {out_ch}), got {out.shape}"
    )


def test_gcn_no_nan() -> None:
    """GCN output should not contain NaN values."""
    model = SimpleGCN(16, 32, 7)
    model.eval()
    x = torch.randn(20, 16)
    edge_index = _make_ring_graph(20)
    out = model(x, edge_index)
    assert not torch.isnan(out).any(), "GCN output contains NaN"


def test_gcn_gradients_flow() -> None:
    """Loss should be differentiable through the GCN."""
    model = SimpleGCN(8, 16, 3)
    x = torch.randn(10, 8)
    edge_index = _make_ring_graph(10)
    labels = torch.randint(0, 3, (10,))
    out = model(x, edge_index)
    loss = F.cross_entropy(out, labels)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
