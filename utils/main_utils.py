import mlflow

from utils.logger import App_Logger
from utils.read_params import read_params

config = read_params()

log_writter = App_Logger()


def log_model_to_mlflow(model, model_name, db_name, collection_name):
    try:
        mlflow.sklearn.log_model(
            sk_model=model,
            serialization_format=config["mlflow_config"]["serialization_format"],
            registered_model_name=model_name,
        )

        log_writter.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Logged {model_name} in mlflow",
        )

    except Exception as e:
        raise e


def log_param_to_mlflow(model, model_name, param_name, db_name, collection_name):
    try:
        mlflow.log_param(
            key=model_name + "-" + param_name, value=model.__dict__[param_name]
        )

        log_writter.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=model_name + f" {param_name} logged in mlflow",
        )

    except Exception as e:
        raise e


def log_metric_to_mlflow(model_name, metric, db_name, collection_name):
    try:
        mlflow.log_metric(key=model_name + "-best_error", value=metric)

        log_writter.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=model_name + "-best error logged in mlflow",
        )

    except Exception as e:
        raise e
