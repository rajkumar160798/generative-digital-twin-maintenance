import os
import numpy as np
import pandas as pd


def _extract_features_from_file(csv_path: str) -> dict:
    """Extract basic statistics from a single cycle CSV file."""
    df = pd.read_csv(csv_path, header=None, skiprows=1, engine="python")
    values = df.values.flatten().astype(float)
    return {
        "rms": np.sqrt(np.mean(values**2)),
        "max": np.max(values),
        "min": np.min(values),
        "std": np.std(values),
    }


def load_xjtu_data(data_dir: str, loads=None) -> pd.DataFrame:
    """Load and preprocess the XJTU-SY bearing dataset.

    Parameters
    ----------
    data_dir : str
        Root directory containing load condition subfolders.
    loads : list[str], optional
        Specific load condition folders to use. If ``None`` all folders in
        ``data_dir`` are processed.

    Returns
    -------
    pandas.DataFrame
        DataFrame with cycle-level statistics and Remaining Useful Life (RUL).
    """
    if loads is None:
        loads = [p for p in os.listdir(data_dir) if not p.startswith(".")]

    records = []
    for load in loads:
        load_path = os.path.join(data_dir, load)
        bearing_folders = [
            f
            for f in os.listdir(load_path)
            if os.path.isdir(os.path.join(load_path, f))
        ]
        for bearing in sorted(bearing_folders):
            bearing_path = os.path.join(load_path, bearing)
            cycle_files = sorted(
                os.listdir(bearing_path), key=lambda x: int(x.split(".")[0])
            )
            total_cycles = len(cycle_files)
            for idx, csv_file in enumerate(cycle_files, start=1):
                csv_path = os.path.join(bearing_path, csv_file)
                features = _extract_features_from_file(csv_path)
                features["cycle"] = idx
                features["RUL"] = total_cycles - idx
                features["bearing"] = bearing
                features["load"] = load
                records.append(features)

    return pd.DataFrame(records)
