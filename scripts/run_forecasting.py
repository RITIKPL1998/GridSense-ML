import mlflow

from src.data.loader import load_raw_data
from src.data.validator import validate_data
from src.features.feature_builder import build_features
from src.data.splitter import time_series_split
from src.models.supervised.forecasting import train_rolling_forecast

from src.models.supervised.forecasting import (
    run_decomposition,
    train_ml_models
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("energy_load_forecasting")


def main():

    print("Loading data...")

    data_path = "data/raw/smart_grid.csv"

    df = load_raw_data(data_path)
    df = validate_data(df)
    df = build_features(df)

    print("Splitting dataset...")

    train_df, val_df, test_df = time_series_split(df)

    print("Running seasonal decomposition...")

    run_decomposition(df)

    print("Training ML forecasting models...")

    results = train_ml_models(train_df, test_df)

    print("\nFinal Model Comparison:")
    print(results)

    print("\nForecasting pipeline completed.")

    print("Running rolling forecast simulation...")

    train_rolling_forecast(train_df, test_df)


if __name__ == "__main__":
    main()