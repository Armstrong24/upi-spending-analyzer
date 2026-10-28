"""AI-Based UPI Spending Analyzer - core logic (no Streamlit code here)."""
from pathlib import Path

import pandas as pd

from . import categorizer, anomaly, insights

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def analyze(statement: pd.DataFrame, model):
    """Full pipeline: categorise -> add merchant/month -> anomalies -> recurring."""
    df = categorizer.categorize(statement, model)
    df = insights.add_columns(df)
    flagged = anomaly.detect_anomalies(df)
    recurring = insights.find_recurring(df)
    return df, flagged, recurring


def load_model():
    train_df = pd.read_csv(DATA / "training_data.csv")
    return categorizer.train(train_df)
