from src.data.loader import load_raw_data
from src.data.validator import validate_data
from src.features.feature_builder import build_features
from src.data.splitter import time_series_split
from src.models.supervised.train import train_regression_models


def main():

    data_path = "data/raw/smart_grid.csv"

    # load + validate
    df = load_raw_data(data_path)
    df = validate_data(df)

    # feature engineering
    df = build_features(df)

    # time series split
    train_df, val_df, test_df = time_series_split(df)

    print("Data loaded and validated successfully")
    print("Feature engineering completed successfully")
    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    

    print("\nRunning BASELINE models...")
    train_regression_models(train_df, val_df)

    print("\nRunning models WITHOUT power_consumption...")
    train_regression_models(train_df, val_df, drop_power=True)
    print("\nRunning models WITHOUT electrical signals...")
    train_regression_models(train_df, val_df, drop_electrical=True)

    print("\nAll experiments completed successfully")


if __name__ == "__main__":
    main()