⚡ Energy ML Project — Smart Grid Machine Learning Platform
A full end-to-end ML platform for smart grid energy data, featuring regression models,
XGBoost/LightGBM forecasting, anomaly detection, and KMeans clustering — all visualized
in an interactive Streamlit dashboard.

📁 Project Structure
energy-ml-project/
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard
│
├── data/
│   └── raw/
│       └── smart_grid.csv      # Raw grid data (not committed)
│
├── models/
│   └── trained/                # Saved .pkl models (not committed)
│
├── reports/                    # Generated CSVs (not committed)
│   ├── model_performance.csv
│   ├── forecast_results_xgboost.csv
│   ├── forecast_results_lightgbm.csv
│   ├── anomaly_results.csv
│   ├── kmeans_cluster.csv
│   └── clustering_results.csv
│
├── scripts/
│   ├── run_training.py         # Train regression models
│   ├── run_forecasting.py      # Train XGBoost/LightGBM forecasts
│   ├── run_anomaly_detection.py
│   └── run_clustering.py       # KMeans clustering experiments
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
│   │   │   ├── train.py        # Ridge, Lasso, ElasticNet, RF, GBM
│   │   │   └── forecasting.py  # XGBoost, LightGBM
│   │   └── unsupervised/
│   │       └── clustering.py   # KMeans (baseline, feature_reduction, pca)
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

🚀 Quick Start
1. Clone the repo
bashgit clone https://github.com/YOUR_USERNAME/energy-ml-project.git
cd energy-ml-project
2. Create virtual environment
bashpython -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
3. Install dependencies
bashpip install -r requirements.txt
4. Add your data
Place smart_grid.csv in data/raw/
5. Run the pipeline
bash# Train regression models (Ridge, Lasso, ElasticNet, RF, GBM)
python scripts/run_training.py

# Train forecasting models (XGBoost, LightGBM)
python scripts/run_forecasting.py

# Run anomaly detection
python scripts/run_anomaly_detection.py

# Run clustering experiments
python scripts/run_clustering.py
6. Launch dashboard
bashstreamlit run dashboard/app.py

📊 Dashboard Pages
PageDescriptionData ExplorerBrowse raw data, time series plots, column statsModel PerformanceCompare regression models — RMSE, R², MAE across 3 experimentsForecastingXGBoost vs LightGBM forecast vs actual, residuals, rolling windowsAnomaly DetectionScatter plot of detected anomalies, distribution statsClusteringKMeans results by mode (baseline/feature_reduction/pca), silhouette scoresReal-Time PredictionEnter grid parameters and get live load predictions

🧪 Models
Regression (Supervised)

Linear Regression, Ridge, Lasso, ElasticNet
Random Forest, Gradient Boosting
3 experiments: Baseline | Drop Power | Drop Electrical

Forecasting

XGBoost, LightGBM
Single-step, multi-step (96-step horizon), rolling forecast

Clustering (Unsupervised)

KMeans with k=2..5
3 modes: Baseline | Feature Reduction | PCA


📦 Requirements
See requirements.txt for full list. Key packages:

streamlit, plotly, pandas, numpy
scikit-learn, xgboost, lightgbm
mlflow, joblib
statsmodels


🔬 MLflow Tracking
All experiments are tracked with MLflow. To view:
bashmlflow ui
Then open http://localhost:5000