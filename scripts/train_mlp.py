"""Train an MLP on the MNIST dataset."""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int,
                 dropout: float = 0.0) -> None:
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


def main(cfg_path: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    root = cfg["data"]["root"]
    train_set = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=cfg["data"]["batch_size"],
                              shuffle=True, num_workers=cfg["data"]["num_workers"])
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    mc = cfg["model"]
    model = MLP(mc["input_dim"], mc["hidden_dims"], mc["output_dim"],
                mc["dropout"]).to(device)

    tc = cfg["training"]
    optimizer = optim.Adam(model.parameters(), lr=tc["lr"],
                           weight_decay=tc["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=tc["epochs"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            if batch_idx % cfg["logging"]["log_interval"] == 0:
                print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_set)}] "
                      f"Loss: {loss.item():.4f}")
        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                correct += model(data).argmax(dim=1).eq(target).sum().item()
        print(f"Epoch {epoch} Test Accuracy: {correct / len(test_set):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on MNIST")
    parser.add_argument("--config", default="configs/mlp_mnist.yaml")
    args = parser.parse_args()
    main(args.config)
