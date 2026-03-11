"""
run_clustering.py
Runs KMeans clustering experiments across 3 modes × k=2..5
and saves all results to reports/
"""

from src.data.loader import load_raw_data
from src.data.validator import validate_data
from src.features.feature_builder import build_features
from src.models.unsupervised.clustering import run_kmeans_clustering

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("energy_load_forecasting")


def main():

    print("Loading data...")
    df = load_raw_data("data/raw/smart_grid.csv")
    df = validate_data(df)
    df = build_features(df)
    print(f"Data shape: {df.shape}")

    modes = ["baseline", "feature_reduction", "pca"]

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"MODE: {mode}")
        print(f"{'='*50}")

        for k in range(2, 6):
            print(f"\n  k = {k}")
            run_kmeans_clustering(df.copy(), n_clusters=k, mode=mode)

    print("\n✅ All clustering experiments completed.")
    print("Results saved to reports/")
    print("  - kmeans_cluster.csv          (master, used by dashboard)")
    print("  - kmeans_cluster_<mode>_k<n>.csv  (per-run files)")
    print("  - clustering_results.csv      (summary of all runs)")


if __name__ == "__main__":
    main()