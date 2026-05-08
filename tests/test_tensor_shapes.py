"""Tests that verify output tensor shapes for core model building blocks."""

import torch
import torch.nn as nn
import pytest


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int],
                 output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_mlp_output_shape(batch_size: int) -> None:
    """MLP output must have shape (batch_size, output_dim)."""
    model = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
    model.eval()
    x = torch.randn(batch_size, 784)
    out = model(x)
    assert out.shape == (batch_size, 10), (
        f"Expected ({batch_size}, 10), got {out.shape}"
    )


def test_mlp_flattens_images() -> None:
    """MLP should accept 2-D image input (B, C*H*W) and flat input alike."""
    model = MLP(input_dim=784, hidden_dims=[128], output_dim=10)
    model.eval()
    # Flat input
    x_flat = torch.randn(4, 784)
    assert model(x_flat).shape == (4, 10)


def test_linear_layer_shape() -> None:
    """Sanity check: a single Linear layer preserves batch dimension."""
    layer = nn.Linear(16, 8)
    x = torch.randn(5, 16)
    assert layer(x).shape == (5, 8)


def test_relu_preserves_shape() -> None:
    """ReLU must not change tensor shape."""
    relu = nn.ReLU()
    x = torch.randn(3, 64)
    assert relu(x).shape == x.shape


def test_sequential_chain_shape() -> None:
    """A Sequential chain of Linear+ReLU layers must produce the correct shape."""
    model = nn.Sequential(
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 8), nn.ReLU(),
        nn.Linear(8, 4),
    )
    x = torch.randn(10, 32)
    assert model(x).shape == (10, 4)
