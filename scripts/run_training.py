from src.data.loader import load_raw_data
from src.data.validator import validate_data
from src.features.feature_builder import build_features
from src.data.splitter import time_series_split
from src.models.supervised.train import train_regression_models

def main():

    data_path = "data/raw/smart_grid.csv"

    df = load_raw_data(data_path)
    df = validate_data(df)
    df = build_features(df)
    train_df, val_df, test_df = time_series_split(df)
    models = train_regression_models(train_df, val_df)
    print("Model training completed successfully")
    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)
    print("Feature engineering completed successfully")
    print("Data loaded and validated successfully")
    print(df.head())
    print(df.shape)


if __name__ == "__main__":
    main()