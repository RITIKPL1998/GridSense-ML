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
def create_laf_features(df: pd.DataFrame) -> pd.DataFrame:
    target = 'predicted_load_kw'

    df["load_lag_1"] = df[target].shift(1)
    df["load_lag_2"] = df[target].shift(2)
    df["load_lag_4"] = df[target].shift(4)
    df["load_lag_8"] = df[target].shift(8)

    return df

def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    target = 'predicted_load_kw'

    df["rolling_mean_4"] = df[target].rolling(window=4).mean()
    df["rolling_std_4"] = df[target].rolling(window=4).std()

    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df = clean_column_names(df)

    df = create_time_features(df)
    df = create_laf_features(df)
    df = create_rolling_features(df)
    df = df.dropna()

    return df

def select_features(df, target, drop_power=False, drop_electrical=False):

    drop_cols = ["timestamp", target]

    if drop_power:
        drop_cols.append("power_consumption_kw")

    if drop_electrical:
        drop_cols.extend([
            "power_consumption_kw",
            "voltage_v",
            "current_a"
        ])

    X = df.drop(columns=drop_cols)
    y = df[target]

    return X, y