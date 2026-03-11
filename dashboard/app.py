import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Grid ML Platform",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Grid Machine Learning Platform")

# --------------------------------------------------
# PATHS  (dashboard/ is one level below project root)
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]   # energy-ml-project/

DATA_PATH     = BASE_DIR / "data"    / "raw"     / "smart_grid.csv"
MODEL_PATH    = BASE_DIR / "models"  / "trained"              # ← FIXED: was "models/"
ANOMALY_PATH  = BASE_DIR / "reports" / "anomaly_results.csv"
CLUSTER_PATH  = BASE_DIR / "reports" / "kmeans_cluster.csv"
FORECAST_PATH = BASE_DIR / "reports" / "forecast_results.csv"
FORECAST_XGB_PATH  = BASE_DIR / "reports" / "forecast_results_xgboost.csv"
FORECAST_LGB_PATH  = BASE_DIR / "reports" / "forecast_results_lightgbm.csv"
# model_results.csv OR model_performance.csv — accept either name
_r1 = BASE_DIR / "reports" / "model_results.csv"
_r2 = BASE_DIR / "reports" / "model_performance.csv"
RESULTS_PATH = _r1 if _r1.exists() else _r2

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("%", "percent", regex=False)
        .str.replace("/", "_per_", regex=False)
    )
    return df


# --------------------------------------------------
# LOAD RAW DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    df = clean_cols(df)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Select Section",
    [
        "Data Explorer",
        "Model Performance",
        "Forecasting",
        "Anomaly Detection",
        "Clustering",
        "Real-Time Prediction",
    ]
)

# --------------------------------------------------
# DATA EXPLORER
# --------------------------------------------------

if page == "Data Explorer":

    st.header("📊 Dataset Overview")

    if df is None:
        st.error(f"Data file not found at: {DATA_PATH}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Time Range",
                f"{df['timestamp'].dt.date.min()} → {df['timestamp'].dt.date.max()}"
                if "timestamp" in df.columns else "N/A")

    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Column Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    # --- time series chart ---
    if "timestamp" in df.columns and "power_consumption_kw" in df.columns:
        st.subheader("Power Consumption Over Time")
        df_plot = df.sort_values("timestamp").iloc[::20]
        fig = px.line(
            df_plot,
            x="timestamp",
            y="power_consumption_kw",
            title="Power Consumption Over Time",
            labels={"power_consumption_kw": "Power (kW)", "timestamp": "Time"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- numeric column selector ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        st.subheader("Explore Any Column Over Time")
        selected_col = st.selectbox("Choose a column to plot", numeric_cols)
        if "timestamp" in df.columns:
            df_plot2 = df.sort_values("timestamp").iloc[::10]
            fig2 = px.line(df_plot2, x="timestamp", y=selected_col,
                           title=f"{selected_col} Over Time")
            st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("🤖 Regression Model Comparison")

    if not RESULTS_PATH.exists():
        st.error(
            f"**Model results file not found.**\n\n"
            f"Expected at: `{RESULTS_PATH}`\n\n"
            "Run `python scripts/run_training.py` first to generate model results."
        )
        st.stop()

    results = pd.read_csv(RESULTS_PATH)

    # ------------------------------------------------------------------
    # DEDUPLICATION
    # train.py saves inside the for-loop so each run appends duplicates.
    # Keep only the LAST result per (model, drop_power, drop_electrical).
    # ------------------------------------------------------------------
    key_cols = ["model", "drop_power", "drop_electrical"]
    available_keys = [c for c in key_cols if c in results.columns]
    if available_keys:
        results = (
            results
            .drop_duplicates(subset=available_keys, keep="last")
            .reset_index(drop=True)
        )

    # --- experiment filter ---
    if "drop_power" in results.columns and "drop_electrical" in results.columns:

        label_map = {
            (False, False): "Baseline (all features)",
            (True,  False): "Without power_consumption",
            (False, True):  "Without electrical signals",
            (True,  True):  "Without electrical signals",
        }

        experiment_options = results[["drop_power", "drop_electrical"]].drop_duplicates()
        filter_labels = [
            label_map.get((bool(r["drop_power"]), bool(r["drop_electrical"])), "Custom")
            for _, r in experiment_options.iterrows()
        ]

        selected_label = st.selectbox("Filter by experiment", filter_labels)
        idx = filter_labels.index(selected_label)
        selected_row = experiment_options.iloc[idx]
        results = results[
            (results["drop_power"] == selected_row["drop_power"]) &
            (results["drop_electrical"] == selected_row["drop_electrical"])
        ].reset_index(drop=True)

    st.subheader("Results Table")
    st.dataframe(results.reset_index(drop=True), use_container_width=True)

    col1, col2, col3 = st.columns(3)

    fig_rmse = px.bar(
        results, x="model", y="rmse",
        title="RMSE (lower is better)",
        color="model", color_discrete_sequence=px.colors.qualitative.Set2,
        text_auto=".3f"
    )
    fig_rmse.update_layout(showlegend=False)
    col1.plotly_chart(fig_rmse, use_container_width=True)

    fig_r2 = px.bar(
        results, x="model", y="r2",
        title="R² Score (higher is better)",
        color="model", color_discrete_sequence=px.colors.qualitative.Set2,
        text_auto=".3f"
    )
    fig_r2.update_layout(showlegend=False)
    col2.plotly_chart(fig_r2, use_container_width=True)

    fig_mae = px.bar(
        results, x="model", y="mae",
        title="MAE (lower is better)",
        color="model", color_discrete_sequence=px.colors.qualitative.Set2,
        text_auto=".3f"
    )
    fig_mae.update_layout(showlegend=False)
    col3.plotly_chart(fig_mae, use_container_width=True)

    # --- best model highlight ---
    best = results.loc[results["r2"].idxmax()]
    st.success(
        f"🏆 **Best model:** `{best['model']}` "
        f"— R² = {best['r2']:.4f}, RMSE = {best['rmse']:.4f}, MAE = {best['mae']:.4f}"
    )

    # --- feature importance CSVs (if present) ---
    importance_files = list(BASE_DIR.glob("*_feature_importance.csv"))
    if importance_files:
        st.subheader("Feature Importance")
        selected_imp = st.selectbox(
            "Select model",
            [f.name for f in importance_files]
        )
        imp_df = pd.read_csv(BASE_DIR / selected_imp)
        fig_imp = px.bar(
            imp_df.head(15), x="importance", y="feature",
            orientation="h", title=f"Top Features — {selected_imp}",
            color="importance", color_continuous_scale="Blues"
        )
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)

# --------------------------------------------------
# FORECASTING
# --------------------------------------------------

elif page == "Forecasting":

    st.header("📈 Energy Load Forecast")

    # ── paths for all forecast-related files ──────────────────────────
    FORECAST_MODEL_RESULTS = BASE_DIR / "reports" / "forecast_model_results.csv"

    # ── tab layout ────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📊 Forecast vs Actual",
        "🏆 Model Comparison",
        "🔁 Rolling Forecast",
    ])

    # ================================================================
    # TAB 1 — Forecast vs Actual  (from forecast_results.csv)
    # ================================================================
    with tab1:

        st.subheader("XGBoost / LightGBM Forecast vs Actual")

        if not FORECAST_PATH.exists():
            st.warning(
                f"No forecast results found at `{FORECAST_PATH}`.\n\n"
                "Run `python scripts/run_forecasting.py` to generate them."
            )

            # ── fallback: live preview from regression models ────────
            if df is not None and MODEL_PATH.exists():
                model_files = list(MODEL_PATH.glob("*.pkl"))
                if model_files and "timestamp" in df.columns:
                    st.info("Showing a live preview using your regression models instead.")

                    # Only show BASELINE models (trained on all features)
                    baseline_stems = None
                    if RESULTS_PATH.exists():
                        res_meta = pd.read_csv(RESULTS_PATH)
                        res_meta = res_meta.drop_duplicates(
                            subset=["model", "drop_power", "drop_electrical"], keep="last"
                        )
                        baseline_models = res_meta[
                            (~res_meta["drop_power"].astype(bool)) &
                            (~res_meta["drop_electrical"].astype(bool))
                        ]["model"].tolist()
                        baseline_stems = []
                        for mf in model_files:
                            for bm in baseline_models:
                                if mf.stem.startswith(bm):
                                    baseline_stems.append(mf.stem)
                                    break

                    display_files = (
                        [f for f in model_files if f.stem in baseline_stems]
                        if baseline_stems else model_files
                    )
                    if not display_files:
                        display_files = model_files

                    model_names = [m.stem for m in display_files]
                    selected = st.selectbox(
                        "Select model for preview (baseline models only)", model_names
                    )
                    model_file = next(f for f in display_files if f.stem == selected)
                    model = joblib.load(model_file)

                    target = "predicted_load_kw"
                    preview_df = df.copy()
                    preview_df["hour"]        = preview_df["timestamp"].dt.hour
                    preview_df["day"]         = preview_df["timestamp"].dt.day
                    preview_df["month"]       = preview_df["timestamp"].dt.month
                    preview_df["day_of_week"] = preview_df["timestamp"].dt.dayofweek
                    preview_df["is_weekend"]  = preview_df["day_of_week"].isin([5, 6]).astype(int)

                    if target in preview_df.columns:
                        preview_df["load_lag_1"]     = preview_df[target].shift(1)
                        preview_df["load_lag_2"]     = preview_df[target].shift(2)
                        preview_df["load_lag_4"]     = preview_df[target].shift(4)
                        preview_df["load_lag_8"]     = preview_df[target].shift(8)
                        preview_df["rolling_mean_4"] = preview_df[target].rolling(4).mean()
                        preview_df["rolling_std_4"]  = preview_df[target].rolling(4).std()
                        preview_df = preview_df.dropna()

                        # Use EXACTLY the features the model was trained on
                        try:
                            expected_features = list(model.feature_names_in_)
                        except AttributeError:
                            expected_features = [
                                c for c in preview_df.columns
                                if c not in ["timestamp", target]
                            ]

                        missing = [f for f in expected_features if f not in preview_df.columns]
                        if missing:
                            st.error(
                                f"Model `{selected}` expects features not found in data: "
                                f"`{'`, `'.join(missing)}`\n\n"
                                "Run `python scripts/run_forecasting.py` for real XGBoost/LightGBM results."
                            )
                        else:
                            try:
                                preds = model.predict(preview_df[expected_features])
                                preview_df["prediction"] = preds
                                preview_df = preview_df.sort_values("timestamp").iloc[::10]

                                fig = px.line(
                                    preview_df, x="timestamp",
                                    y=[target, "prediction"],
                                    title=f"Actual vs Predicted — {selected}",
                                    labels={
                                        target: "Actual Load (kW)",
                                        "prediction": "Predicted (kW)"
                                    },
                                    color_discrete_map={
                                        target: "#1f77b4",
                                        "prediction": "#ff7f0e"
                                    },
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                rmse_prev = np.sqrt(np.mean(
                                    (preview_df[target] - preview_df["prediction"])**2
                                ))
                                mae_prev = (
                                    preview_df[target] - preview_df["prediction"]
                                ).abs().mean()
                                c1, c2 = st.columns(2)
                                c1.metric("Preview RMSE", f"{rmse_prev:.4f} kW")
                                c2.metric("Preview MAE",  f"{mae_prev:.4f} kW")
                            except Exception as e:
                                st.error(f"Could not generate preview: {e}")

        else:
            # ── model selector: per-model CSVs take priority ─────────
            available_models = {}
            if FORECAST_XGB_PATH.exists():
                available_models["XGBoost"] = FORECAST_XGB_PATH
            if FORECAST_LGB_PATH.exists():
                available_models["LightGBM"] = FORECAST_LGB_PATH
            if not available_models and FORECAST_PATH.exists():
                available_models["LightGBM (legacy)"] = FORECAST_PATH

            selected_fc_model = st.radio(
                "Select forecast model to view",
                list(available_models.keys()),
                horizontal=True
            )
            chosen_path = available_models[selected_fc_model]

            forecast = pd.read_csv(chosen_path)
            forecast = clean_cols(forecast)
            forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])

            actual_col = next((c for c in forecast.columns if "actual" in c), None)
            pred_col   = next((c for c in forecast.columns if "pred"   in c), None)

            if not actual_col or not pred_col:
                st.warning("Could not detect actual/prediction columns. Showing raw data.")
                st.dataframe(forecast, use_container_width=True)
            else:
                # ── date range filter ────────────────────────────────
                min_date = forecast["timestamp"].min().date()
                max_date = forecast["timestamp"].max().date()
                col_a, col_b = st.columns(2)
                start_date = col_a.date_input("From", min_date,
                                              min_value=min_date, max_value=max_date)
                end_date   = col_b.date_input("To",   max_date,
                                              min_value=min_date, max_value=max_date)

                mask = (
                    (forecast["timestamp"].dt.date >= start_date) &
                    (forecast["timestamp"].dt.date <= end_date)
                )
                filt = forecast[mask].copy()

                if filt.empty:
                    st.warning("No data in selected date range.")
                else:
                    fig = px.line(
                        filt, x="timestamp",
                        y=[actual_col, pred_col],
                        title=f"{selected_fc_model} — Forecast vs Actual Load",
                        labels={"value": "Load (kW)", "variable": "Series",
                                "timestamp": "Time"},
                        color_discrete_map={
                            actual_col: "#1f77b4",
                            pred_col:   "#ff7f0e"
                        },
                    )
                    fig.update_layout(legend_title_text="")
                    st.plotly_chart(fig, use_container_width=True)

                    filt["residual"] = filt[actual_col] - filt[pred_col]
                    rmse = np.sqrt(np.mean(filt["residual"]**2))
                    mae  = filt["residual"].abs().mean()
                    eps  = 1e-10
                    mape = np.mean(
                        np.abs(filt["residual"] / (filt[actual_col] + eps))
                    ) * 100

                    c1, c2, c3 = st.columns(3)
                    c1.metric("RMSE",  f"{rmse:.4f} kW")
                    c2.metric("MAE",   f"{mae:.4f} kW")
                    c3.metric("MAPE",  f"{mape:.2f}%")

                    st.subheader("Residuals Over Time")
                    fig_res = px.line(
                        filt, x="timestamp", y="residual",
                        title="Prediction Error (Actual − Predicted)",
                        labels={"residual": "Residual (kW)"},
                        color_discrete_sequence=["#d62728"],
                    )
                    fig_res.add_hline(y=0, line_dash="dash", line_color="grey")
                    st.plotly_chart(fig_res, use_container_width=True)

                    col1, col2 = st.columns(2)
                    fig_hist = px.histogram(
                        filt, x="residual", nbins=60,
                        title="Residual Distribution",
                        color_discrete_sequence=["#9467bd"],
                    )
                    col1.plotly_chart(fig_hist, use_container_width=True)

                    fig_scat = px.scatter(
                        filt, x=actual_col, y=pred_col,
                        title="Actual vs Predicted (scatter)",
                        labels={actual_col: "Actual (kW)", pred_col: "Predicted (kW)"},
                        opacity=0.4,
                        color_discrete_sequence=["#2ca02c"],
                    )
                    mn = filt[[actual_col, pred_col]].min().min()
                    mx = filt[[actual_col, pred_col]].max().max()
                    fig_scat.add_trace(go.Scatter(
                        x=[mn, mx], y=[mn, mx],
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        name="Perfect fit"
                    ))
                    col2.plotly_chart(fig_scat, use_container_width=True)

    # ================================================================
    # TAB 2 — Model Comparison  (from forecast_model_results.csv)
    # ================================================================
    with tab2:

        st.subheader("XGBoost vs LightGBM — Forecast Model Comparison")

        if not FORECAST_MODEL_RESULTS.exists():
            st.warning(
                f"No forecast model comparison found at `{FORECAST_MODEL_RESULTS}`.\n\n"
                "Run `python scripts/run_forecasting.py` to generate it."
            )
        else:
            fm = pd.read_csv(FORECAST_MODEL_RESULTS)
            fm = clean_cols(fm)

            st.dataframe(fm, use_container_width=True)

            metric_cols = [c for c in ["rmse", "mae", "mape"] if c in fm.columns]

            for metric in metric_cols:
                fig_m = px.bar(
                    fm, x="model", y=metric,
                    title=f"{metric.upper()} by Model",
                    color="model",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    text_auto=".3f",
                )
                fig_m.update_layout(showlegend=False)
                st.plotly_chart(fig_m, use_container_width=True)

            # best model
            if "rmse" in fm.columns:
                best_fc = fm.loc[fm["rmse"].idxmin()]
                st.success(
                    f"🏆 **Best forecast model:** `{best_fc['model']}` "
                    f"— RMSE = {best_fc['rmse']:.4f} kW"
                    + (f", MAPE = {best_fc['mape']:.2f}%" if "mape" in best_fc else "")
                )

    # ================================================================
    # TAB 3 — Rolling Forecast horizon explorer
    # ================================================================
    with tab3:

        st.subheader("96-Step Rolling Forecast (24-hour horizon)")

        # ── model selector for rolling tab ──────────────────────────
        roll_models = {}
        if FORECAST_XGB_PATH.exists():
            roll_models["XGBoost"] = FORECAST_XGB_PATH
        if FORECAST_LGB_PATH.exists():
            roll_models["LightGBM"] = FORECAST_LGB_PATH
        if not roll_models and FORECAST_PATH.exists():
            roll_models["LightGBM (legacy)"] = FORECAST_PATH

        if not roll_models:
            st.warning(
                "Run `python scripts/run_forecasting.py` first to generate forecast data."
            )
        else:
            selected_roll_model = st.radio(
                "Select model", list(roll_models.keys()), horizontal=True
            )
            forecast = pd.read_csv(roll_models[selected_roll_model])
            forecast = clean_cols(forecast)
            forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])

            actual_col = next((c for c in forecast.columns if "actual" in c), None)
            pred_col   = next((c for c in forecast.columns if "pred"   in c), None)

            if actual_col and pred_col:
                # slice into 96-step windows and let user pick one
                n_windows = len(forecast) // 96
                if n_windows == 0:
                    st.info("Not enough rows for a 96-step window view.")
                else:
                    window_idx = st.slider(
                        "Select 96-step forecast window (each = 24 hrs at 15-min intervals)",
                        1, n_windows, 1
                    )
                    start = (window_idx - 1) * 96
                    window = forecast.iloc[start: start + 96].copy()

                    fig_roll = px.line(
                        window, x="timestamp",
                        y=[actual_col, pred_col],
                        title=f"Window {window_idx}: 96-step forecast",
                        labels={"value": "Load (kW)", "variable": "Series",
                                "timestamp": "Time"},
                        color_discrete_map={
                            actual_col: "#1f77b4",
                            pred_col:   "#ff7f0e"
                        },
                    )
                    st.plotly_chart(fig_roll, use_container_width=True)

                    window["residual"] = window[actual_col] - window[pred_col]
                    rmse_w = np.sqrt(np.mean(window["residual"]**2))
                    mae_w  = window["residual"].abs().mean()
                    c1, c2 = st.columns(2)
                    c1.metric("Window RMSE", f"{rmse_w:.4f} kW")
                    c2.metric("Window MAE",  f"{mae_w:.4f} kW")

                    # error per window heatmap
                    st.subheader("RMSE Across All Windows")
                    window_rmses = []
                    for i in range(n_windows):
                        s = i * 96
                        w = forecast.iloc[s: s + 96]
                        err = np.sqrt(np.mean((w[actual_col] - w[pred_col])**2))
                        window_rmses.append({"window": i + 1, "rmse": err})
                    wdf = pd.DataFrame(window_rmses)
                    fig_wrmse = px.bar(
                        wdf, x="window", y="rmse",
                        title="RMSE per 96-step Window",
                        labels={"window": "Window #", "rmse": "RMSE (kW)"},
                        color="rmse", color_continuous_scale="RdYlGn_r",
                    )
                    st.plotly_chart(fig_wrmse, use_container_width=True)
            else:
                st.info("Forecast file found but actual/prediction columns not detected.")

# --------------------------------------------------
# ANOMALY DETECTION
# --------------------------------------------------

elif page == "Anomaly Detection":

    st.header("🚨 Grid Anomaly Detection")

    if not ANOMALY_PATH.exists():
        st.error(
            f"Anomaly results not found at `{ANOMALY_PATH}`.\n\n"
            "Run `python scripts/run_anomaly_detection.py` first."
        )
        st.stop()

    anomaly_df = pd.read_csv(ANOMALY_PATH)
    anomaly_df = clean_cols(anomaly_df)

    if "timestamp" in anomaly_df.columns:
        anomaly_df["timestamp"] = pd.to_datetime(anomaly_df["timestamp"])

    # Auto-detect anomaly column
    flag_col = next(
        (c for c in anomaly_df.columns if "anomaly" in c or "flag" in c or "label" in c),
        None
    )
    y_col = next(
        (c for c in anomaly_df.columns if "power" in c or "load" in c or "consumption" in c),
        anomaly_df.select_dtypes("number").columns[0] if len(anomaly_df.select_dtypes("number").columns) else None
    )

    if flag_col is None:
        st.warning("Could not find an anomaly flag column. Showing raw data.")
        st.dataframe(anomaly_df, use_container_width=True)
        st.stop()

    # Summary metrics
    total      = len(anomaly_df)
    n_anomaly  = (anomaly_df[flag_col] != 0).sum() if anomaly_df[flag_col].dtype != object \
                 else (anomaly_df[flag_col] == "anomaly").sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{total:,}")
    col2.metric("Anomalies Detected", f"{n_anomaly:,}")
    col3.metric("Anomaly Rate", f"{n_anomaly/total*100:.2f}%")

    # Scatter chart
    x_col = "timestamp" if "timestamp" in anomaly_df.columns else anomaly_df.columns[0]
    fig = px.scatter(
        anomaly_df,
        x=x_col,
        y=y_col,
        color=flag_col,
        title="Detected Grid Anomalies",
        color_discrete_map={-1: "red", 0: "steelblue", 1: "red",
                            "anomaly": "red", "normal": "steelblue"},
        labels={y_col: y_col.replace("_", " ").title(), flag_col: "Status"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomaly Distribution")
    dist = anomaly_df[flag_col].value_counts().reset_index()
    dist.columns = ["label", "count"]
    fig2 = px.pie(dist, names="label", values="count",
                  title="Normal vs Anomaly Split")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Anomaly Records")
    anomaly_rows = anomaly_df[anomaly_df[flag_col] != 0] \
        if anomaly_df[flag_col].dtype != object \
        else anomaly_df[anomaly_df[flag_col] == "anomaly"]
    st.dataframe(anomaly_rows, use_container_width=True)

# --------------------------------------------------
# CLUSTERING
# --------------------------------------------------

elif page == "Clustering":

    st.header("🔵 Grid Behaviour Clustering")

    CLUSTERING_SUMMARY = BASE_DIR / "reports" / "clustering_results.csv"
    REPORTS_DIR        = BASE_DIR / "reports"

    # ── check if ANY cluster file exists ─────────────────────────────
    per_run_files = list(REPORTS_DIR.glob("kmeans_cluster_*.csv")) if REPORTS_DIR.exists() else []
    master_exists = CLUSTER_PATH.exists()

    if not per_run_files and not master_exists:
        st.error(
            f"No cluster results found in `{REPORTS_DIR}`\n\n"
            "Run `python scripts/run_clustering.py` first."
        )
        st.stop()

    # ── tabs ─────────────────────────────────────────────────────────
    tab_c1, tab_c2, tab_c3 = st.tabs([
        "🔬 Explore by Mode & K",
        "📊 Mode Comparison",
        "🏆 Silhouette Scores",
    ])

    # ================================================================
    # TAB C1 — interactive explorer: pick mode + k
    # ================================================================
    with tab_c1:

        st.subheader("Explore Cluster Results")

        # build list of available (mode, k) from per-run files
        available = {}
        for f in sorted(per_run_files):
            # filename: kmeans_cluster_{mode}_k{n}.csv
            stem = f.stem  # e.g. kmeans_cluster_baseline_k3
            parts = stem.replace("kmeans_cluster_", "").rsplit("_k", 1)
            if len(parts) == 2:
                mode_name, k_str = parts
                try:
                    k_val = int(k_str)
                    available.setdefault(mode_name, []).append(k_val)
                except ValueError:
                    pass

        if not available:
            # fall back to master file
            if master_exists:
                available_modes = ["master"]
            else:
                st.warning("No per-run cluster files found.")
                st.stop()
        else:
            available_modes = list(available.keys())

        col_m, col_k = st.columns(2)
        selected_mode = col_m.selectbox(
            "Clustering Mode",
            available_modes,
            format_func=lambda x: {
                "baseline":          "Baseline (electrical features)",
                "feature_reduction": "Feature Reduction (energy + env)",
                "pca":               "PCA (dimensionality reduced)",
                "master":            "Master (all modes combined)",
            }.get(x, x)
        )

        if selected_mode == "master":
            cluster_df = pd.read_csv(CLUSTER_PATH)
            k_options  = None
        else:
            k_options = sorted(available.get(selected_mode, []))
            selected_k = col_k.selectbox("Number of Clusters (k)", k_options)
            run_file   = REPORTS_DIR / f"kmeans_cluster_{selected_mode}_k{selected_k}.csv"
            cluster_df = pd.read_csv(run_file)

        cluster_df  = clean_cols(cluster_df)
        cluster_col = "cluster"

        if cluster_col not in cluster_df.columns:
            st.warning("No 'cluster' column found in this file.")
            st.dataframe(cluster_df.head(), use_container_width=True)
            st.stop()

        cluster_df[cluster_col] = cluster_df[cluster_col].astype(str)

        numeric_cols = cluster_df.select_dtypes("number").columns.tolist()

        # ── summary metrics ───────────────────────────────────────────
        n_k = cluster_df[cluster_col].nunique()
        c1, c2 = st.columns(2)
        c1.metric("Total Records", f"{len(cluster_df):,}")
        c2.metric("Clusters", n_k)

        # ── axis picker ───────────────────────────────────────────────
        default_x = next((c for c in ["power_consumption_kw", "voltage_v", numeric_cols[0]] if c in numeric_cols), numeric_cols[0])
        default_y = next((c for c in ["current_a", "reactive_power_kvar", numeric_cols[1] if len(numeric_cols)>1 else numeric_cols[0]] if c in numeric_cols), numeric_cols[min(1,len(numeric_cols)-1)])

        col_a, col_b = st.columns(2)
        x_axis = col_a.selectbox("X axis", numeric_cols, index=numeric_cols.index(default_x) if default_x in numeric_cols else 0)
        y_axis = col_b.selectbox("Y axis", numeric_cols, index=numeric_cols.index(default_y) if default_y in numeric_cols else min(1, len(numeric_cols)-1))

        title_label = f"{selected_mode.replace('_',' ').title()} — k={selected_k}" if k_options else "All Modes"
        fig = px.scatter(
            cluster_df, x=x_axis, y=y_axis,
            color=cluster_col,
            color_discrete_sequence=px.colors.qualitative.Set1,
            title=f"Clusters: {title_label}  |  {x_axis} vs {y_axis}",
            labels={x_axis: x_axis.replace("_"," ").title(),
                    y_axis: y_axis.replace("_"," ").title()},
            opacity=0.65,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── distribution bar ──────────────────────────────────────────
        col_d, col_m2 = st.columns(2)
        dist = cluster_df[cluster_col].value_counts().reset_index()
        dist.columns = ["Cluster", "Count"]
        fig_dist = px.bar(dist, x="Cluster", y="Count",
                          color="Cluster", title="Records per Cluster",
                          color_discrete_sequence=px.colors.qualitative.Set1,
                          text_auto=True)
        col_d.plotly_chart(fig_dist, use_container_width=True)

        fig_pie = px.pie(dist, names="Cluster", values="Count",
                         title="Cluster Share",
                         color_discrete_sequence=px.colors.qualitative.Set1)
        col_m2.plotly_chart(fig_pie, use_container_width=True)

        # ── cluster means table ───────────────────────────────────────
        st.subheader("Cluster Feature Means")
        means = cluster_df.groupby(cluster_col)[numeric_cols].mean().round(3)
        st.dataframe(means, use_container_width=True)

    # ================================================================
    # TAB C2 — side-by-side mode comparison (same k)
    # ================================================================
    with tab_c2:

        st.subheader("Compare Modes at Same k")

        if not per_run_files:
            st.info("Per-run cluster files not found. Run clustering first.")
        else:
            compare_k = st.slider("Select k to compare across modes", 2, 5, 4)
            modes_to_compare = ["baseline", "feature_reduction", "pca"]

            cols = st.columns(len(modes_to_compare))

            for col, mode_name in zip(cols, modes_to_compare):
                f_path = REPORTS_DIR / f"kmeans_cluster_{mode_name}_k{compare_k}.csv"
                if f_path.exists():
                    cdf = clean_cols(pd.read_csv(f_path))
                    cdf["cluster"] = cdf["cluster"].astype(str)
                    num_c = cdf.select_dtypes("number").columns.tolist()
                    xc = "power_consumption_kw" if "power_consumption_kw" in num_c else num_c[0]
                    yc = "reactive_power_kvar"  if "reactive_power_kvar"  in num_c else (num_c[1] if len(num_c)>1 else num_c[0])
                    fig_c = px.scatter(
                        cdf, x=xc, y=yc, color="cluster",
                        title=f"{mode_name.replace('_',' ').title()}\nk={compare_k}",
                        color_discrete_sequence=px.colors.qualitative.Set1,
                        opacity=0.6,
                    )
                    fig_c.update_layout(showlegend=False, height=350,
                                        margin=dict(t=60, b=20, l=20, r=20))
                    col.plotly_chart(fig_c, use_container_width=True)
                else:
                    col.info(f"No file for {mode_name} k={compare_k}")

    # ================================================================
    # TAB C3 — silhouette score comparison across all runs
    # ================================================================
    with tab_c3:

        st.subheader("Silhouette Score Across All Experiments")

        if not CLUSTERING_SUMMARY.exists():
            st.info("clustering_results.csv not found. Run clustering first.")
        else:
            summary = pd.read_csv(CLUSTERING_SUMMARY)
            summary = summary.drop_duplicates(subset=["mode", "n_clusters"], keep="last")

            fig_sil = px.line(
                summary, x="n_clusters", y="silhouette_score",
                color="mode", markers=True,
                title="Silhouette Score by k and Mode",
                labels={"n_clusters": "Number of Clusters (k)",
                        "silhouette_score": "Silhouette Score",
                        "mode": "Mode"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_sil.update_layout(legend_title_text="Mode")
            st.plotly_chart(fig_sil, use_container_width=True)

            st.subheader("Best k per Mode")
            best_per_mode = (
                summary.loc[summary.groupby("mode")["silhouette_score"].idxmax()]
                .sort_values("silhouette_score", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(best_per_mode, use_container_width=True)

            best_overall = best_per_mode.iloc[0]
            st.success(
                f"🏆 **Best overall:** Mode=`{best_overall['mode']}`, "
                f"k={int(best_overall['n_clusters'])}, "
                f"Silhouette={best_overall['silhouette_score']:.4f}"
            )

            st.subheader("All Experiments")
            st.dataframe(
                summary.sort_values(["mode","n_clusters"]).reset_index(drop=True),
                use_container_width=True
            )

# --------------------------------------------------
# REAL-TIME PREDICTION
# --------------------------------------------------

elif page == "Real-Time Prediction":

    st.header("🔮 Predict Energy Load")

    # ── locate models ──────────────────────────────
    model_files = list(MODEL_PATH.glob("*.pkl")) if MODEL_PATH.exists() else []

    # fallback: try root models/ folder (old location)
    if not model_files:
        old_model_path = BASE_DIR / "models"
        model_files = list(old_model_path.glob("*.pkl"))

    if not model_files:
        st.error(
            f"No `.pkl` models found.\n\n"
            f"Checked:\n- `{MODEL_PATH}`\n- `{BASE_DIR / 'models'}`\n\n"
            "Run `python scripts/run_training.py` to train and save models."
        )
        st.stop()

    model_names = [m.stem for m in model_files]
    selected_model_name = st.selectbox("Select Model", model_names)
    model_file = next(f for f in model_files if f.stem == selected_model_name)
    model = joblib.load(model_file)

    # ── get expected feature count ─────────────────
    n_features_expected = None
    try:
        n_features_expected = model.n_features_in_
    except AttributeError:
        pass

    st.subheader("Enter Grid Parameters")

    # Raw sensor inputs (always available)
    col1, col2 = st.columns(2)
    voltage = col1.number_input("Voltage (V)",           200.0, 260.0, 230.0)
    current = col2.number_input("Current (A)",             0.0, 100.0,  20.0)
    solar   = col1.number_input("Solar Power (kW)",        0.0, 100.0,  10.0)
    wind    = col2.number_input("Wind Power (kW)",         0.0, 100.0,   5.0)
    grid_s  = col1.number_input("Grid Supply (kW)",        0.0, 100.0,   1.0)
    power   = col2.number_input("Power Consumption (kW)",  0.0, 500.0,  50.0)

    st.subheader("Time Context")
    col3, col4 = st.columns(2)
    hour        = col3.slider("Hour of Day",      0, 23, 12)
    day_of_week = col4.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
    month       = col3.slider("Month",            1, 12,  6)
    day         = col4.slider("Day of Month",     1, 31, 15)
    is_weekend  = 1 if day_of_week >= 5 else 0

    st.subheader("Recent Load Context (for lag/rolling features)")
    col5, col6 = st.columns(2)
    lag1 = col5.number_input("Load 15 min ago (kW)",  0.0, 1000.0, 50.0)
    lag2 = col6.number_input("Load 30 min ago (kW)",  0.0, 1000.0, 48.0)
    lag4 = col5.number_input("Load 1 hr ago (kW)",    0.0, 1000.0, 45.0)
    lag8 = col6.number_input("Load 2 hrs ago (kW)",   0.0, 1000.0, 42.0)
    roll_mean = col5.number_input("Rolling Mean 4 (kW)", 0.0, 1000.0, 47.0)
    roll_std  = col6.number_input("Rolling Std 4 (kW)",  0.0, 200.0,   3.0)

    if st.button("⚡ Predict Load", type="primary"):

        # Build full feature set matching feature_builder.py output
        feature_dict = {
            "voltage_v":            voltage,
            "current_a":            current,
            "solar_power_kw":       solar,
            "wind_power_kw":        wind,
            "grid_supply_kw":       grid_s,
            "power_consumption_kw": power,
            "hour":                 hour,
            "day":                  day,
            "month":                month,
            "day_of_week":          day_of_week,
            "is_weekend":           is_weekend,
            "load_lag_1":           lag1,
            "load_lag_2":           lag2,
            "load_lag_4":           lag4,
            "load_lag_8":           lag8,
            "rolling_mean_4":       roll_mean,
            "rolling_std_4":        roll_std,
        }

        features = pd.DataFrame([feature_dict])

        # If model was trained without power/electrical, trim columns
        if n_features_expected is not None and n_features_expected < len(features.columns):
            # Try dropping power column first
            try_drop = features.drop(columns=["power_consumption_kw"], errors="ignore")
            if len(try_drop.columns) == n_features_expected:
                features = try_drop
            else:
                try_drop2 = try_drop.drop(
                    columns=["voltage_v", "current_a"], errors="ignore"
                )
                if len(try_drop2.columns) == n_features_expected:
                    features = try_drop2

        try:
            prediction = model.predict(features)[0]
            st.success(f"### Predicted Load: **{prediction:.2f} kW**")

            # Show input summary
            with st.expander("Input features used"):
                st.dataframe(features.T.rename(columns={0: "value"}),
                             use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info(
                f"Model expects **{n_features_expected}** features. "
                f"We provided **{len(features.columns)}**.\n\n"
                f"Features provided: {list(features.columns)}"
            )