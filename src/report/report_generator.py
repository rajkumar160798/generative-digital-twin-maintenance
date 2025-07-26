import pandas as pd


def create_report(
    data: pd.DataFrame, faults, explanations: pd.DataFrame, rul_predictions, schedule
):
    """Assemble a summary report of all pipeline stages."""
    report = {
        "data_summary": data.describe().to_dict(),
        "num_faults": len(faults),
        "top_explanations": explanations.head().to_dict(orient="records"),
        "rul_stats": {
            "min": float(min(rul_predictions)),
            "max": float(max(rul_predictions)),
        },
        "maintenance_schedule": schedule,
    }
    return report
