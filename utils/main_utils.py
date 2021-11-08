import mlflow
import yaml


def read_params(config_path="params.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


config = read_params()


def log_model_to_mlflow(model, model_name):
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=config["mlflow_config"]["s3_bucket"],
        serialization_format=config["mlflow_config"]["seriliazation_format"],
        registered_model_name=model_name,
    )
