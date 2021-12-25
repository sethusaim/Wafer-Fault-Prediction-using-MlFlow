import mlflow
from utils.logger import App_Logger
from utils.read_params import read_params


class Mlflow_Operations:
    def __init__(self, db_name, collection_name):
        self.config = read_params()

        self.log_writter = App_Logger()

        self.db_name = db_name

        self.collection_name = collection_name

    def log_model(self, model, model_name, db_name, collection_name):
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                serialization_format=self.config["mlflow_config"][
                    "serialization_format"
                ],
                registered_model_name=model_name,
                artifact_path=model_name,
            )

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Logged {model_name} in mlflow",
            )

        except Exception as e:
            raise Exception(
                "Exception occured in main_utils..py, Method : log_model, Error : ",
                str(e),
            )

    def log_param(self, model, model_name, param_name, db_name, collection_name):
        try:
            mlflow.log_param(
                key=model_name + "-" + param_name, value=model.__dict__[param_name]
            )

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=model_name + f" {param_name} logged in mlflow",
            )

        except Exception as e:
            raise Exception(
                "Exception occured in main_utils.py, Method : log_param, Error : ",
                str(e),
            )

    def log_metric(self, model_name, metric, db_name, collection_name):
        try:
            mlflow.log_metric(key=model_name + "-best_score", value=metric)

            self.log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=model_name + "-best score logged in mlflow",
            )

        except Exception as e:
            raise Exception(
                "Exception occured in main_utils.py, Method : log_metric, Error : ",
                str(e),
            )

    def log_xgboost_params(self, model):
        try:
            self.log_param(
                model=model,
                model_name=self.config["model_names"]["xgb_model_name"],
                param_name="learning_rate",
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.log_param(
                model=model,
                model_name=self.config["model_names"]["xgb_model_name"],
                param_name="max_depth",
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.log_param(
                model=model,
                model_name=self.config["model_names"]["xgb_model_name"],
                param_name="n_estimators",
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Mlflow_Operations, Method : log_xgboost_params, Error : {str(e)}",
            )

    def log_random_forest_params(self, model):
        try:
            self.log_param(
                model=model,
                model_name=self.config["model_names"]["rf_model_name"],
                param_name="max_depth",
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.log_param(
                model=model,
                model_name=self.config["model_names"]["rf_model_name"],
                param_name="n_estimators",
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.log_param(
                model=model,
                model_name=self.config["model_names"]["rf_model_name"],
                param_name="criterion",
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.log_param(
                model=model,
                model_name=self.config["model_names"]["rf_model_name"],
                param_name="max_features",
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

        except Exception as e:
            raise e

    def log_trained_models(self, kmeans_model, xgboost_model, random_forest_model):
        try:
            self.log_model(
                model=kmeans_model,
                model_name=self.config["model_names"]["kmeans_model_name"],
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.log_model(
                model=xgboost_model,
                model_name=self.config["model_names"]["xgb_model_name"],
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.log_model(
                model=random_forest_model,
                model_name=self.config["model_names"]["rf_model_name"],
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

        except Exception as e:
            raise e

    def log_metrics_of_trained_models(self, xgb_score, rf_score):
        try:
            self.log_metric(
                model_name=self.config["model_names"]["xgb_model_name"],
                metric=float(xgb_score),
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.log_metric(
                model_name=self.config["model_names"]["rf_model_name"],
                metric=float(rf_score),
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

        except Exception as e:
            raise e
