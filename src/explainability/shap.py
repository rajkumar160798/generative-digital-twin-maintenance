import numpy as np
import pandas as pd
import shap
import torch
from torch.utils.data import DataLoader

from src.models.lstm import RULDataset, LSTMModel


def run_shap_analysis(df: pd.DataFrame, model: LSTMModel | None = None):
    """Compute global SHAP values for sensor importance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RULDataset(df)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    if model is None:
        model = LSTMModel(len(sensor_cols)).to(device)
        model.eval()

    batch = next(iter(loader))[0].to(device)
    explainer = shap.GradientExplainer(model, batch)
    shap_values = explainer.shap_values(batch)[0]
    shap_summary = np.mean(np.abs(shap_values), axis=(0, 1))
    shap_df = pd.DataFrame({"sensor": sensor_cols, "importance": shap_summary})
    shap_df.sort_values(by="importance", ascending=False, inplace=True)
    return shap_df


def analyze_drift_over_rul(df: pd.DataFrame, bins: int = 5):
    """Analyze SHAP value drift over different RUL ranges."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    df["RUL_bin"] = pd.cut(df["RUL"], bins, labels=False)
    shap_by_bin: dict[int, list[np.ndarray]] = {i: [] for i in range(bins)}

    for bin_idx in range(bins):
        bin_df = df[df["RUL_bin"] == bin_idx]
        if bin_df.empty:
            continue
        shap_df = run_shap_analysis(bin_df)
        shap_by_bin[bin_idx].append(shap_df["importance"].values)

    avg_importance = {
        b: np.mean(vals, axis=0) if vals else np.zeros(len(sensor_cols))
        for b, vals in shap_by_bin.items()
    }
    drift_df = pd.DataFrame(avg_importance, index=sensor_cols).T
    return drift_df
