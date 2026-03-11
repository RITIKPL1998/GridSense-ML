from pathlib import Path
import pandas as pd


def load_raw_data(data_path: str) -> pd.DataFrame:

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Parse timestamp column
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    return df