import os
import pandas as pd
import numpy as np


def load_cmapss_data(data_dir: str) -> pd.DataFrame:
    """Load NASA C-MAPSS FD004 dataset and compute RUL."""
    column_names = [
        "unit",
        "time",
        "operational_setting_1",
        "operational_setting_2",
        "operational_setting_3",
    ] + [f"sensor_{i}" for i in range(1, 22)]

    train_path = os.path.join(data_dir, "train_FD004.txt")
    df = pd.read_csv(train_path, sep=" ", header=None)
    df.drop([26, 27], axis=1, inplace=True)
    df.columns = column_names

    rul_df = df.groupby("unit")["time"].max().reset_index()
    rul_df.columns = ["unit", "max_time"]
    df = df.merge(rul_df, on="unit")
    df["RUL"] = df["max_time"] - df["time"]
    df.drop(columns="max_time", inplace=True)

    return df
