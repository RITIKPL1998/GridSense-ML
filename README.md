# ⚡ GridSense-ML — Smart Grid Intelligence Platform

> An end-to-end Machine Learning platform for smart grid energy analytics — featuring
> predictive load forecasting, anomaly detection, behavioural clustering, and a live
> interactive dashboard.

---

## 🚀 Live App

👉 https://gridsense-ml-rit.streamlit.app/

---

## 🏭 Industry Relevance

Designed with real-world energy industry challenges in mind:

| Challenge                     | Solution in GridSense-ML                            |
| ----------------------------- | --------------------------------------------------- |
| Grid load prediction          | XGBoost / LightGBM forecasting with 96-step horizon |
| Fault & anomaly detection     | Isolation Forest anomaly detection                  |
| Grid behaviour segmentation   | KMeans clustering (3 modes, k=2–5)                  |
| Model comparison & governance | MLflow experiment tracking                          |
| Operational dashboard         | Streamlit real-time prediction interface            |

---

## 📁 Project Structure

```
GridSense-ML/
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── raw/
│       └── smart_grid.csv
│
├── models/
│   └── trained/
│
├── reports/
│   ├── model_performance.csv
│   ├── forecast_results_xgboost.csv
│   ├── forecast_results_lightgbm.csv
│   ├── forecast_model_results.csv
│   ├── anomaly_results.csv
│   ├── kmeans_cluster.csv
│   └── clustering_results.csv
│
├── scripts/
│   ├── run_training.py
│   ├── run_forecasting.py
│   ├── run_anomaly_detection.py
│   └── run_clustering.py
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── mlflow_tracking/
│   └── visualization/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/GridSense-ML.git
cd GridSense-ML

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Place `smart_grid.csv` in `data/raw/`

Run pipelines:

```bash
python scripts/run_training.py
python scripts/run_forecasting.py
python scripts/run_anomaly_detection.py
python scripts/run_clustering.py
```

Run dashboard:

```bash
streamlit run dashboard/app.py
```

---

## 📊 Dashboard Pages

* Data Explorer
* Model Performance
* Forecasting
* Anomaly Detection
* Clustering
* Real-Time Prediction

---

## 🧪 Models & Experiments

### Regression

Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting

### Forecasting

XGBoost, LightGBM (single, multi-step, rolling)

### Clustering

KMeans (baseline, feature reduction, PCA)

---

## 🔬 MLflow

```bash
mlflow ui
```

---

## 📦 Tech Stack

* pandas, numpy
* scikit-learn, xgboost, lightgbm
* statsmodels
* MLflow
* Streamlit, Plotly
* joblib
