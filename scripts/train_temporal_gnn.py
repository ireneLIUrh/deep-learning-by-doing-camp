"""Train a Temporal GNN on a synthetic toy dataset."""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGNN(nn.Module):
    """Simple recurrent GNN: per-step linear message passing + GRU cell."""

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.msg = nn.Linear(in_channels, hidden_channels)
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (T, N, F) — T timesteps, N nodes, F features
        Returns:
            out: (T, N, out_channels)
        """
        T, N, _ = x_seq.shape
        h = torch.zeros(N, self.gru.hidden_size, device=x_seq.device)
        outputs = []
        for t in range(T):
            m = F.relu(self.msg(x_seq[t]))
            m = F.dropout(m, p=self.dropout, training=self.training)
            h = self.gru(m, h)
            outputs.append(self.out(h))
        return torch.stack(outputs)


def make_toy_data(num_nodes: int, num_timesteps: int, num_features: int,
                  seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic temporal node features and regression targets."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    x_seq = torch.randn(num_timesteps, num_nodes, num_features, generator=rng)
    y_seq = x_seq.mean(dim=-1, keepdim=True).expand(
        num_timesteps, num_nodes, num_features)
    return x_seq, y_seq


def main(cfg_path: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dc = cfg["data"]
    x_seq, y_seq = make_toy_data(dc["num_nodes"], dc["num_timesteps"],
                                 dc["num_features"])
    x_seq, y_seq = x_seq.to(device), y_seq.to(device)

    mc = cfg["model"]
    model = TemporalGNN(mc["in_channels"], mc["hidden_channels"],
                        mc["out_channels"], mc["dropout"]).to(device)

    tc = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"],
                                 weight_decay=tc["weight_decay"])

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_seq)
        loss = F.mse_loss(pred, y_seq)
        loss.backward()
        optimizer.step()

        if epoch % cfg["logging"]["log_interval"] == 0:
            print(f"Epoch {epoch:4d} | MSE Loss: {loss.item():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Temporal GNN on toy data")
    parser.add_argument("--config", default="configs/temporal_gnn_toy.yaml")
    args = parser.parse_args()
    main(args.config)
