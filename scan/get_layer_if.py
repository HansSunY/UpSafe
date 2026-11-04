"""SS-Score scan, step 2/3: per-layer linear probes (the SS-Score).

Trains a binary classifier (linear + sigmoid, BCE, Adam lr=1e-3, 50 epochs,
80/20 split) on every ``layers.<i>.mlp.pt`` tensor and records the best
validation loss ("SS-Score") per layer into ``res.log``.
Lower SS-Score == the layer's representations separate harmful/benign better.

Usage:
  python scan/get_layer_if.py scan/features_llama-8b
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


set_seed(42)

HERE = Path(__file__).parent


class BinaryDataset(Dataset):
    def __init__(self, file_path, csv=None):
        csv = csv or str(HERE / "merged_dataset.csv")
        self.data = pd.read_csv(csv)
        self.features = torch.load(file_path)
        assert len(self.data) == self.features.shape[0], (
            "feature tensor and probe csv row count mismatch — regenerate features "
            "with the same csv"
        )
        self.labels = self.data["label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx, ...].to(dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, layer, log_path, epochs):
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x).squeeze(-1)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x).squeeze(-1)
                loss = criterion(outputs, y)
                val_loss += loss.item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}")

    print(f"Lowest Validation Loss: {best_val_loss / len(val_loader):.4f}")
    with open(log_path / "res.log", "a+") as f:
        print(json.dumps({"layer_name": layer, "val_loss": best_val_loss / len(val_loader)}),
              flush=True, file=f)
    return model


def run_main(file_path: str = "", csv: str = ""):
    dataset = BinaryDataset(file_path, csv=csv)
    log_path = Path(file_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_size = dataset.features.shape[1]
    model = LinearModel(input_size=input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer,
                               log_path.stem, log_path.parent, epochs=50)


def run_batch(file_path: str = "", csv: str = ""):
    """Run the probe over every *.pt in a directory (default: scan/features_*)."""
    pts = sorted(Path(file_path).glob("*.pt"))
    if not pts:
        raise ValueError(f"No feature tensors found in {file_path}")
    (Path(file_path) / "res.log").write_text("")
    for pt in tqdm(pts, total=len(pts)):
        run_main(str(pt), csv=csv)


if __name__ == "__main__":
    import fire
    fire.Fire(run_batch)
