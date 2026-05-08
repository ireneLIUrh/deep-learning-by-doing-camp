"""Train a GCN on the Cora citation dataset."""

import argparse
import yaml
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def main(cfg_path: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Planetoid(root=cfg["data"]["root"], name=cfg["data"]["dataset"])
    data = dataset[0].to(device)

    mc = cfg["model"]
    model = GCN(mc["in_channels"], mc["hidden_channels"], mc["out_channels"],
                mc["num_layers"], mc["dropout"]).to(device)

    tc = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"],
                                 weight_decay=tc["weight_decay"])

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % cfg["logging"]["log_interval"] == 0:
            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                acc = pred[data.test_mask].eq(data.y[data.test_mask]).float().mean()
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
                  f"Test Acc: {acc.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GCN on Cora")
    parser.add_argument("--config", default="configs/gcn_cora.yaml")
    args = parser.parse_args()
    main(args.config)
