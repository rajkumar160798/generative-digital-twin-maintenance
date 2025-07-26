import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd


class RULDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int = 50):
        self.sequences = []
        self.targets = []
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        for unit in df["unit"].unique():
            unit_df = df[df["unit"] == unit].reset_index(drop=True)
            for i in range(len(unit_df) - window_size + 1):
                window = unit_df.loc[i : i + window_size - 1, sensor_cols].values
                self.sequences.append(window)
                self.targets.append(unit_df.loc[i + window_size - 1, "RUL"])
        self.sequences = torch.tensor(self.sequences, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


def predict_rul(df: pd.DataFrame, epochs: int = 15, window_size: int = 50):
    """Train an LSTM on the provided dataframe and return RUL predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RULDataset(df, window_size)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    input_size = dataset.sequences.shape[-1]

    model = LSTMModel(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        predictions = model(dataset.sequences.to(device)).cpu().numpy().flatten()
    return predictions
