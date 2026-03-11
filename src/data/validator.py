import pandas as pd


EXPECTED_COLUMNS = [
    "Timestamp",
    "Voltage (V)",
    "Current (A)",
    "Power Consumption (kW)",
    "Reactive Power (kVAR)",
    "Power Factor",
    "Solar Power (kW)",
    "Wind Power (kW)",
    "Grid Supply (kW)",
    "Voltage Fluctuation (%)",
    "Overload Condition",
    "Transformer Fault",
    "Temperature (°C)",
    "Humidity (%)",
    "Electricity Price (USD/kWh)",
    "Predicted Load (kW)",
]


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates dataset before it enters ML pipeline
    """

    # 1. Check columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # 2. Check duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Warning: {duplicates} duplicate rows found")

    # 3. Check missing values
    missing = df.isnull().sum()

    if missing.sum() > 0:
        print("Missing values detected:")
        print(missing[missing > 0])

    # 4. Check timestamp ordering
    if not df["Timestamp"].is_monotonic_increasing:
        print("Warning: timestamps are not ordered")

    # 5. Basic sanity checks
    if (df["Voltage (V)"] < 0).any():
        raise ValueError("Voltage cannot be negative")

    if (df["Current (A)"] < 0).any():
        raise ValueError("Current cannot be negative")

    return df