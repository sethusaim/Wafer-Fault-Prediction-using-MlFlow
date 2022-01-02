import mlflow
from utils.exception import raise_exception
from utils.logger import App_Logger
from utils.main_utils import get_model_name
from utils.read_params import read_params


class Mlflow_Operations:
    def __init__(self, db_name, collection_name):
        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.log_writer = App_Logger()

        self.db_name = db_name

        self.collection_name = collection_name

    def log_model(self, model, model_name, db_name, collection_name):
        method_name = self.log_model.__name__

        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                serialization_format=self.config["mlflow_config"][
                    "serialization_format"
                ],
                registered_model_name=model_name,
                artifact_path=model_name,
            )

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Logged {model_name} in mlflow",
            )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def log_metric(self, model_name, metric, db_name, collection_name):
        method_name = self.log_metric.__name__

        try:
            mlflow.log_metric(key=model_name + "-best_score", value=metric)

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=model_name + "-best score logged in mlflow",
            )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def log_param(self, idx, model, model_name, param, db_name, collection_name):
        method_name = self.log_param.__name__

        try:
            name = model_name + str(idx) + f"-{param}"

            mlflow.log_param(key=name, value=model.__dict__[param])

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"{name} logged in mlflow",
            )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def log_xgboost_params(self, idx, model, db_name, collection_name):
        method_name = self.log_xgboost_params.__name__

        try:
            model_name = get_model_name(
                model, db_name=db_name, collection_name=collection_name
            )

            params_list = list(self.config["model_params"]["xgb_model"].keys())

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message="Got xgboost params list",
            )

            for param in params_list:
                self.log_param(
                    idx=idx,
                    model=model,
                    model_name=model_name,
                    param=param,
                    db_name=db_name,
                    collection_name=collection_name,
                )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def log_rf_model_params(self, idx, model, db_name, collection_name):
        method_name = self.log_rf_model_params.__name__

        try:
            model_name = get_model_name(
                model=model, db_name=db_name, collection_name=collection_name
            )

            params_list = list(self.config["model_params"]["rf_model"].keys())

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message="Got rf model params",
            )

            for param in params_list:
                self.log_param(
                    idx=idx,
                    model=model,
                    model_name=model_name,
                    param=param,
                    db_name=db_name,
                    collection_name=collection_name,
                )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def log_trained_models(
        self, kmeans_model, idx, xgb_model, rf_model, db_name, collection_name
    ):
        method_name = self.log_trained_models.__name__

        try:
            xgb_model_name = get_model_name(
                model=xgb_model, db_name=db_name, collection_name=collection_name
            )

            rf_model_name = get_model_name(
                model=rf_model, db_name=db_name, collection_name=collection_name
            )

            kmeans_model_name = get_model_name(
                model=kmeans_model, db_name=db_name, collection_name=collection_name
            )

            self.log_model(
                model=kmeans_model,
                model_name=kmeans_model_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.log_model(
                model=xgb_model,
                model_name=xgb_model_name + str(idx),
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.log_model(
                model=rf_model,
                model_name=rf_model_name + str(idx),
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

    def log_metrics_of_trained_models(
        self, idx, xgb_model, rf_model, xgb_score, rf_score, db_name, collection_name
    ):
        method_name = self.log_metrics_of_trained_models.__name__

        try:
            self.log_metric(
                model_name=xgb_model.__class__.__name__ + str(idx),
                metric=float(xgb_score),
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.log_metric(
                model_name=rf_model.__class__.__name__ + str(idx),
                metric=float(rf_score),
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message="Metrics for trained models are looged in mlflow",
            )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )
