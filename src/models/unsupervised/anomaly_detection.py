import pandas as pd
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("energy_load_forecasting")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def run_isolation_forest(df, use_grid_features=False):

    if use_grid_features:

        feature_set = [
            "voltage_v",
            "current_a",
            "reactive_power_kvar",
            "power_factor",
            "voltage_fluctuation_percent",
            "temperature_°c"
        ]

        run_name = "isolation_forest_grid_features"

    else:

        feature_set = df.drop(columns=[
            "timestamp",
            "predicted_load_kw",
            "transformer_fault",
            "overload_condition"
        ]).columns

        run_name = "isolation_forest_all_features"

    X = df[feature_set]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42
    )

    with mlflow.start_run(run_name=run_name):

        model.fit(X_scaled)

        anomaly_scores = model.decision_function(X_scaled)
        anomalies = model.predict(X_scaled)

        df["anomaly_score"] = anomaly_scores
        df["anomaly_flag"] = anomalies

        mlflow.log_param("model", "IsolationForest")
        mlflow.log_param("feature_mode", "grid_features" if use_grid_features else "all_features")
        mlflow.log_param("contamination", 0.01)
        mlflow.log_param("n_estimators", 200)
        mlflow.log_metric("num_anomalies", (df["anomaly_flag"] == -1).sum())

        fault_overlap = df[(df["anomaly_flag"] == -1) & (df["transformer_fault"] == 1)].shape[0]

        mlflow.log_metric("fault_overlap", fault_overlap)

    return df