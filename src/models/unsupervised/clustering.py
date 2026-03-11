import pandas as pd
import numpy as np
import mlflow

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("energy_load_forecasting")


# ─────────────────────────────────────────────
# Feature sets
# ─────────────────────────────────────────────

BASELINE_FEATURES = [
    "voltage_v",
    "current_a",
    "power_consumption_kw",
    "reactive_power_kvar",
    "power_factor",
    "solar_power_kw",
    "wind_power_kw",
    "grid_supply_kw",
]

REDUCED_FEATURES = [
    "power_consumption_kw",
    "reactive_power_kvar",
    "power_factor",
    "solar_power_kw",
    "wind_power_kw",
    "grid_supply_kw",
    "voltage_fluctuation_percent",
    "temperature_c",          # column cleaned name (° → removed by feature_builder)
]


def _available_features(df, feature_list):
    """Return only features that actually exist in df."""
    return [f for f in feature_list if f in df.columns]


def run_kmeans_clustering(df, n_clusters=4, mode="baseline"):
    """
    Run KMeans clustering, save results to reports/kmeans_cluster_{mode}_k{n}.csv
    and append a summary row to reports/clustering_results.csv.

    Returns df with a 'cluster' column added.
    """

    if mode == "baseline":
        features = _available_features(df, BASELINE_FEATURES)
    elif mode in ("feature_reduction", "pca"):
        features = _available_features(df, REDUCED_FEATURES)
    else:
        raise ValueError(f"Unknown clustering mode: {mode}")

    if not features:
        print(f"[WARN] No features available for mode={mode}. Skipping.")
        return df

    X = df[features].copy()

    # ── scaling ──────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── PCA (only for pca mode) ───────────────────────────────────────
    n_components = None
    if mode == "pca":
        n_components = min(3, X_scaled.shape[1])
        pca      = PCA(n_components=n_components)
        X_scaled = pca.fit_transform(X_scaled)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA explained variance ({n_components} components): {explained:.3f}")

    # ── KMeans ───────────────────────────────────────────────────────
    model    = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)

    # ── silhouette score ─────────────────────────────────────────────
    if n_clusters > 1:
        sil_score = silhouette_score(X_scaled, clusters, sample_size=min(5000, len(X_scaled)))
    else:
        sil_score = 0.0

    df = df.copy()
    df["cluster"] = clusters
    df["cluster_mode"] = mode

    print(f"  KMeans mode={mode} k={n_clusters} | silhouette={sil_score:.4f}")
    print(f"  Cluster distribution:\n{df['cluster'].value_counts().to_string()}")

    # ── MLflow logging ───────────────────────────────────────────────
    with mlflow.start_run(run_name=f"kmeans_{mode}_k{n_clusters}"):
        mlflow.log_param("model",        "KMeans")
        mlflow.log_param("mode",         mode)
        mlflow.log_param("n_clusters",   n_clusters)
        mlflow.log_param("num_features", len(features))
        if n_components:
            mlflow.log_param("pca_components", n_components)

        mlflow.log_metric("silhouette_score", sil_score)

        for c, count in df["cluster"].value_counts().items():
            mlflow.log_metric(f"cluster_{c}_size", int(count))

    # ── save per-run CSV ─────────────────────────────────────────────
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Full cluster assignment (all original cols + cluster label)
    out_cols = ["timestamp"] if "timestamp" in df.columns else []
    out_cols += features + ["cluster", "cluster_mode"]
    cluster_df = df[out_cols].copy()

    per_run_path = reports_dir / f"kmeans_cluster_{mode}_k{n_clusters}.csv"
    cluster_df.to_csv(per_run_path, index=False)
    print(f"  Saved {per_run_path}")

    # ── also save/update the master kmeans_cluster.csv ───────────────
    # (used by dashboard — keeps the BEST silhouette run per mode)
    master_path  = reports_dir / "kmeans_cluster.csv"
    summary_path = reports_dir / "clustering_results.csv"

    # append to summary
    summary_row = pd.DataFrame([{
        "mode":              mode,
        "n_clusters":        n_clusters,
        "silhouette_score":  sil_score,
        "n_features":        len(features),
        "file":              str(per_run_path.name),
    }])
    if summary_path.exists():
        existing_summary = pd.read_csv(summary_path)
        summary_row = pd.concat([existing_summary, summary_row], ignore_index=True)
        summary_row = summary_row.drop_duplicates(
            subset=["mode", "n_clusters"], keep="last"
        )
    summary_row.to_csv(summary_path, index=False)

    # update master if this run has the best silhouette for this mode
    if master_path.exists():
        current_master = pd.read_csv(master_path)
        current_modes  = current_master["cluster_mode"].unique() if "cluster_mode" in current_master.columns else []
        if mode in current_modes:
            current_sil = silhouette_score(
                X_scaled, clusters, sample_size=min(5000, len(X_scaled))
            ) if n_clusters > 1 else 0.0
            # replace master content for this mode if better
            other_modes = current_master[current_master["cluster_mode"] != mode]
            master_df   = pd.concat([other_modes, cluster_df], ignore_index=True)
        else:
            master_df = pd.concat([current_master, cluster_df], ignore_index=True)
    else:
        master_df = cluster_df

    master_df.to_csv(master_path, index=False)

    return df