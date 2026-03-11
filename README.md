# ⚡ GridSense-ML — Smart Grid Intelligence Platform

> An end-to-end Machine Learning platform for smart grid energy analytics — featuring 
> predictive load forecasting, anomaly detection, behavioural clustering, and a live 
> interactive dashboard. Built to demonstrate production-grade ML engineering on real 
> electrical grid data.

---

## 🏭 Industry Relevance

Designed with real-world energy industry challenges in mind:

| Challenge | Solution in GridSense-ML |
|-----------|--------------------------|
| Grid load prediction | XGBoost / LightGBM forecasting with 96-step horizon |
| Fault & anomaly detection | Isolation Forest anomaly detection |
| Grid behaviour segmentation | KMeans clustering (3 modes, k=2–5) |
| Model comparison & governance | MLflow experiment tracking |
| Operational dashboard | Streamlit real-time prediction interface |

---

## 📁 Project Structure

```
GridSense-ML/
│
├── dashboard/
│   └── app.py                    # Streamlit dashboard (6 pages)
│
├── data/
│   └── raw/
│       └── smart_grid.csv        # Raw grid sensor data
│
├── models/
│   └── trained/                  # Serialised .pkl models
│
├── reports/                      # Generated experiment outputs
│   ├── model_performance.csv
│   ├── forecast_results_xgboost.csv
│   ├── forecast_results_lightgbm.csv
│   ├── forecast_model_results.csv
│   ├── anomaly_results.csv
│   ├── kmeans_cluster.csv
│   └── clustering_results.csv
│
├── scripts/
│   ├── run_training.py           # Train regression models
│   ├── run_forecasting.py        # Train XGBoost/LightGBM forecasts
│   ├── run_anomaly_detection.py  # Anomaly detection pipeline
│   └── run_clustering.py        # KMeans clustering experiments
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── validator.py
│   │   └── splitter.py
│   ├── features/
│   │   └── feature_builder.py
│   ├── models/
│   │   ├── supervised/
│   │   │   ├── train.py          # Ridge, Lasso, ElasticNet, RF, GBM
│   │   │   └── forecasting.py    # XGBoost, LightGBM (single/multi/rolling)
│   │   └── unsupervised/
│   │       └── clustering.py     # KMeans (baseline, feature_reduction, pca)
│   ├── evaluation/
│   │   └── reports.py
│   ├── mlflow_tracking/
│   │   └── mlflow_logger.py
│   └── visualization/
│       ├── cluster_plots.py
│       └── elbow_plot.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/GridSense-ML.git
cd GridSense-ML
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your data
Place `smart_grid.csv` in `data/raw/`

### 5. Run the full pipeline
```bash
# 1. Regression models (Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting)
python scripts/run_training.py

# 2. Forecasting (XGBoost, LightGBM — single step, multi-step, rolling)
python scripts/run_forecasting.py

# 3. Anomaly detection
python scripts/run_anomaly_detection.py

# 4. Clustering experiments (baseline, feature reduction, PCA — k=2 to 5)
python scripts/run_clustering.py
```

### 6. Launch dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📊 Dashboard Pages

| Page | What You See |
|------|-------------|
| **Data Explorer** | Raw data browser, time series plots, column statistics |
| **Model Performance** | RMSE / R² / MAE comparison across 3 experiments, feature importance |
| **Forecasting** | XGBoost vs LightGBM — forecast vs actual, residuals, 96-step rolling windows |
| **Anomaly Detection** | Scatter plot of grid anomalies, anomaly rate, flagged records |
| **Clustering** | KMeans by mode & k, side-by-side mode comparison, silhouette score analysis |
| **Real-Time Prediction** | Live load prediction — enter grid parameters, get instant output |

---

## 🧪 Models & Experiments

### Supervised — Regression
| Model | Experiments |
|-------|-------------|
| Linear Regression | Baseline, Drop Power, Drop Electrical |
| Ridge | Baseline, Drop Power, Drop Electrical |
| Lasso | Baseline, Drop Power, Drop Electrical |
| ElasticNet | Baseline, Drop Power, Drop Electrical |
| Random Forest | Baseline, Drop Power, Drop Electrical |
| Gradient Boosting | Baseline, Drop Power, Drop Electrical |

### Supervised — Forecasting
| Model | Strategy |
|-------|----------|
| XGBoost | Single-step, 96-step multi-step, rolling window |
| LightGBM | Single-step, 96-step multi-step, rolling window |

### Unsupervised — Clustering
| Mode | Features Used | k Range |
|------|--------------|---------|
| Baseline | Electrical signals (voltage, current, power) | 2–5 |
| Feature Reduction | Energy + environmental features | 2–5 |
| PCA | Dimensionality-reduced features | 2–5 |

---

## 🔬 MLflow Experiment Tracking

All model runs are tracked with MLflow — parameters, metrics, and artifacts.

```bash
mlflow ui
# Open http://localhost:5000
```

---

## 📦 Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | pandas, numpy |
| ML Models | scikit-learn, xgboost, lightgbm |
| Time Series | statsmodels |
| Experiment Tracking | MLflow |
| Dashboard | Streamlit, Plotly |
| Model Serialisation | joblib |

---

## 👤 Author

Built as a demonstration of production-grade ML engineering applied to smart grid energy systems.  
Open to collaboration with energy industry partners — Hitachi Energy, Siemens, ABB, and beyond.
GitHub: @RITIKPL1998
