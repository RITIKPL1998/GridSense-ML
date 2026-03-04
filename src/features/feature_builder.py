import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:

    df.columns = (
        df.columns
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("%", "percent")
        .str.replace("/", "_per_")
    )

    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df = clean_column_names(df)

    df = create_time_features(df)

    return df