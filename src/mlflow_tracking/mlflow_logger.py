import mlflow
import mlflow.sklearn


def start_experiment(experiment_name="energy_forecasting"):

    mlflow.set_experiment(experiment_name)

    return mlflow.start_run()


def log_params(params: dict):

    mlflow.log_params(params)


def log_metrics(metrics: dict):

    mlflow.log_metrics(metrics)


def log_model(model, artifact_path="model"):

    mlflow.sklearn.log_model(model, artifact_path)

    