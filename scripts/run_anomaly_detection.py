from src.data.loader import load_raw_data
from src.data.validator import validate_data
from src.features.feature_builder import build_features
from src.models.unsupervised.anomaly_detection import run_isolation_forest


def main():

    data_path = "data/raw/smart_grid.csv"

    df = load_raw_data(data_path)
    df = validate_data(df)
    df = build_features(df)

    print("\nRunning anomaly detection with ALL features")
    df_all = run_isolation_forest(df.copy(), use_grid_features=False)

    print("\nRunning anomaly detection with GRID features")
    df_grid = run_isolation_forest(df.copy(), use_grid_features=True)


if __name__ == "__main__":
    main()