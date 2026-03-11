import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path


# -----------------------------
# evaluation
# -----------------------------

def evaluate_forecast(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    eps  = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    return rmse, mae, mape


# -----------------------------
# visualization
# -----------------------------

def plot_forecast(y_true, y_pred, title):
    plt.figure(figsize=(14, 6))
    plt.plot(y_true.index, y_true.values, label="Actual", linewidth=2)
    plt.plot(y_true.index[:len(y_pred)], y_pred, label="Forecast", linewidth=2)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), f"{title}.png")
    plt.close()


# -----------------------------
# decomposition
# -----------------------------

def run_decomposition(df):
    series = df.set_index("timestamp")["predicted_load_kw"].asfreq("15min")
    with mlflow.start_run(run_name="seasonal_decomposition"):
        result = seasonal_decompose(series, model="additive", period=96)
        fig = result.plot()
        mlflow.log_figure(fig, "seasonal_decomposition.png")


# -----------------------------
# ML forecasting models
# ── FIX: saves a separate CSV per model so the dashboard
#         can show XGBoost and LightGBM independently
# -----------------------------

def train_ml_models(train_df, test_df):

    target  = "predicted_load_kw"
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    X_train = train_df.drop(columns=[target, "timestamp"])
    y_train = train_df[target]
    X_test  = test_df.drop(columns=[target, "timestamp"])
    y_test  = test_df[target]

    models = {
        "xgboost": XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        ),
    }

    results = []

    for name, model in models.items():

        with mlflow.start_run(run_name=f"{name}_forecast"):

            print(f"Training {name}...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse, mae, mape = evaluate_forecast(y_test, preds)

            mlflow.log_param("model", name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae",  mae)
            mlflow.log_metric("mape", mape)

            plot_forecast(
                y_test.set_axis(range(len(y_test))),
                preds,
                f"{name}_forecast"
            )

            results.append({"model": name, "rmse": rmse, "mae": mae, "mape": mape})

            # ── save per-model forecast CSV ───────────────────────────
            forecast_df = pd.DataFrame({
                "timestamp":  test_df["timestamp"].values,
                "actual":     y_test.values,
                "prediction": preds,
            })
            per_model_path = reports_dir / f"forecast_results_{name}.csv"
            forecast_df.to_csv(per_model_path, index=False)
            print(f"Saved {per_model_path}")

    results_df = pd.DataFrame(results)
    print("\nForecasting Results:")
    print(results_df)

    # save model comparison
    results_df.to_csv(reports_dir / "forecast_model_results.csv", index=False)

    # ── also keep legacy forecast_results.csv (last model = lightgbm) ─
    # so old code that references it still works
    forecast_df.to_csv(reports_dir / "forecast_results.csv", index=False)

    print("All forecast CSVs saved to reports/")
    return results_df


# -----------------------------
# multi-step forecasting
# -----------------------------

def multi_step_forecast(model, last_row, horizon, feature_cols, target_col):
    forecasts   = []
    current_row = last_row.copy()
    for step in range(horizon):
        X    = current_row[feature_cols].values.reshape(1, -1)
        pred = model.predict(X)[0]
        forecasts.append(pred)
        current_row["load_lag_8"] = current_row["load_lag_4"]
        current_row["load_lag_4"] = current_row["load_lag_2"]
        current_row["load_lag_2"] = current_row["load_lag_1"]
        current_row["load_lag_1"] = pred
    return np.array(forecasts)


def train_multi_step_models(train_df, test_df):

    target   = "predicted_load_kw"
    X_train  = train_df.drop(columns=[target, "timestamp"])
    y_train  = train_df[target]
    feature_cols = X_train.columns
    horizon  = 96

    models = {
        "xgboost_multi": XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        ),
        "lightgbm_multi": LGBMRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        ),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}")
            model.fit(X_train, y_train)
            last_row = test_df.iloc[0].drop(["timestamp", target])
            preds    = multi_step_forecast(model, last_row, horizon, feature_cols, target)
            actual   = test_df[target].iloc[:horizon]
            rmse, mae, mape = evaluate_forecast(actual, preds)
            mlflow.log_param("model", name)
            mlflow.log_param("forecast_horizon", horizon)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae",  mae)
            mlflow.log_metric("mape", mape)
            plot_forecast(actual, preds, f"{name}_96_step_forecast")
            print(f"{name} RMSE: {rmse:.4f}")


# -----------------------------
# rolling forecast
# -----------------------------

def rolling_forecast_simulation(model, train_df, test_df, horizon=96):
    target       = "predicted_load_kw"
    feature_cols = train_df.drop(columns=[target, "timestamp"]).columns
    predictions  = []
    actuals      = []
    for start in range(0, len(test_df) - horizon, horizon):
        window   = test_df.iloc[start: start + horizon]
        last_row = window.iloc[0].drop(["timestamp", target])
        preds    = multi_step_forecast(model, last_row, horizon, feature_cols, target)
        predictions.extend(preds)
        actuals.extend(window[target].values[:horizon])
    return np.array(actuals), np.array(predictions)


def train_rolling_forecast(train_df, test_df):

    target  = "predicted_load_kw"
    X_train = train_df.drop(columns=[target, "timestamp"])
    y_train = train_df[target]

    models = {
        "xgboost_rolling": XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        ),
        "lightgbm_rolling": LGBMRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        ),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}")
            model.fit(X_train, y_train)
            actual, preds = rolling_forecast_simulation(model, train_df, test_df)
            rmse, mae, mape = evaluate_forecast(actual, preds)
            mlflow.log_param("model", name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae",  mae)
            mlflow.log_metric("mape", mape)
            plot_forecast(pd.Series(actual), preds, f"{name}_rolling_forecast")
            print(f"{name} RMSE: {rmse:.4f}")