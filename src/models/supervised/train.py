import pandas as pd
import numpy as np
import joblib
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("energy_load_forecasting")

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from pathlib import Path

from src.mlflow_tracking.mlflow_logger import log_params, log_metrics, log_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from src.evaluation.reports import get_feature_importance
from src.features.feature_builder import select_features


def evaluate_model(model, X_val, y_val):

    predictions = model.predict(X_val)

    mse  = mean_squared_error(y_val, predictions)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_val, predictions)
    r2   = r2_score(y_val, predictions)

    return rmse, r2, mae


def train_regression_models(train_df, val_df, drop_power=False, drop_electrical=False):

    target = "predicted_load_kw"
    results = []

    X_train, y_train = select_features(train_df, target, drop_power=drop_power, drop_electrical=drop_electrical)
    X_val,   y_val   = select_features(val_df,   target, drop_power=drop_power, drop_electrical=drop_electrical)

    models = {
        "linear_regression":  LinearRegression(),
        "ridge":              Ridge(alpha=1.0),
        "lasso":              Lasso(alpha=0.01),
        "elastic_net":        ElasticNet(alpha=0.01, l1_ratio=0.5),
        "random_forest":      RandomForestRegressor(
                                  n_estimators=200,
                                  max_depth=10,
                                  random_state=42,
                                  n_jobs=-1
                              ),
        "gradient_boosting":  GradientBoostingRegressor(
                                  n_estimators=200,
                                  learning_rate=0.05,
                                  max_depth=3,
                                  random_state=42
                              ),
    }

    trained_models = {}

    model_dir = Path("models/trained")
    model_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # TRAINING LOOP  –  only collect results here, save CSV AFTER loop
    # ─────────────────────────────────────────────────────────────────
    for name, model in models.items():

        print(f"Training {name}...")

        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)

            rmse, r2, mae = evaluate_model(model, X_val, y_val)

            results.append({
                "model":            name,
                "rmse":             rmse,
                "r2":               r2,
                "mae":              mae,
                "drop_power":       drop_power,
                "drop_electrical":  drop_electrical,
            })

            if name in ["random_forest", "gradient_boosting"]:
                importance_df   = get_feature_importance(model, X_train.columns)
                importance_path = f"{name}_feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)

                print("\nTop Important Features:")
                print(importance_df.head(10))

            print(f"{name} - Validation RMSE: {rmse:.4f}, R2: {r2:.4f}")

            log_params({
                "model_name":                name,
                "drop_power_feature":        drop_power,
                "drop_electrical_features":  drop_electrical,
                "n_features":                X_train.shape[1],
                "train_samples":             len(X_train),
            })

            log_metrics({"rmse": rmse, "r2": r2, "mae": mae})
            log_model(model)

            # save model locally
            model_path = model_dir / f"{name}_model.pkl"
            joblib.dump(model, model_path)
            print(f"Saved model to {model_path}")

        trained_models[name] = model

    # ─────────────────────────────────────────────────────────────────
    # SAVE RESULTS  –  outside the loop so it runs only ONCE
    # Also deduplicate so re-running training doesn't bloat the CSV.
    # ─────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    reports_dir  = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    results_path = reports_dir / "model_results.csv"

    if results_path.exists():
        existing   = pd.read_csv(results_path)
        combined   = pd.concat([existing, results_df], ignore_index=True)
        # Keep only the latest result for each (model, experiment) combination
        combined   = (
            combined
            .drop_duplicates(
                subset=["model", "drop_power", "drop_electrical"],
                keep="last"
            )
            .reset_index(drop=True)
        )
        combined.to_csv(results_path, index=False)
    else:
        results_df.to_csv(results_path, index=False)

    print(f"\nSaved model results to {results_path}")
    print(results_df.to_string(index=False))

    return trained_models