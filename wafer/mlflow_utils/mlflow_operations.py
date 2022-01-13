import mlflow
from mlflow.tracking import MlflowClient

from utils.logger import App_Logger
from utils.main_utils import get_model_name
from utils.read_params import read_params
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class Mlflow_Operations:
    def __init__(self, db_name, collection_name):
        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.log_writer = App_Logger()

        self.s3_obj = S3_Operations()

        self.db_name = db_name

        self.collection_name = collection_name

        self.mlflow_save_format = self.config["mlflow_config"]["serialization_format"]

        self.remote_server_uri = self.config["mlflow_config"]["remote_server_uri"]

        self.trained_models_dir = self.config["models_dir"]["trained"]

        self.staged_models_dir = self.config["models_dir"]["stag"]

        self.prod_models_dir = self.config["models_dir"]["prod"]

        self.model_save_format = self.config["model_params"]["save_format"]

    def get_experiment_from_mlflow(self, exp_name):
        method_name = self.get_experiment_from_mlflow.__name__

        try:
            exp = mlflow.get_experiment_by_name(name=exp_name)

            return exp

        except Exception as e:
            self.log_writer.self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def get_runs_from_mlflow(self, exp_id):
        method_name = self.get_runs_from_mlflow.__name__

        try:
            runs = mlflow.search_runs(experiment_ids=exp_id)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Completed searchiing for runs in mlflow with experiment ids as {exp_id}",
            )

            return runs

        except Exception as e:
            self.log_writer.self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def set_mlflow_experiment(self, experiment_name):
        method_name = self.set_mlflow_experiment.__name__

        try:
            mlflow.set_experiment(experiment_name=experiment_name)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Set mlflow experiment with name as {experiment_name}",
            )

        except Exception as e:
            self.log_writer.self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def get_mlflow_client(self, server_uri):
        method_name = self.get_mlflow_client.__name__

        try:
            client = MlflowClient(tracking_uri=server_uri)

            return client

        except Exception as e:
            self.log_writer.self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def set_mlflow_tracking_uri(self, server_uri):
        method_name = self.set_mlflow_tracking_uri.__name__

        try:
            mlflow.set_tracking_uri(server_uri)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Set mlflow tracking uri to {server_uri}",
            )

        except Exception as e:
            self.log_writer.self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def get_mlflow_models(self):
        method_name = self.get_mlflow_models.__name__

        try:
            client = self.get_mlflow_client(server_uri=self.remote_server_uri)

            reg_model_names = [rm.name for rm in client.list_registered_models()]

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Got registered model from mlflow",
            )

            return reg_model_names

        except Exception as e:
            self.log_writer.self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def search_mlflow_models(self, order):
        method_name = self.search_mlflow_models.__name__

        try:
            client = self.get_mlflow_client(server_uri=self.remote_server_uri)

            results = client.search_registered_models(order_by=[f"name {order}"])

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Got registered models in mlflow in {order} order",
            )

            return results

        except Exception as e:
            self.log_writer.self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def log_model(self, model, model_name):
        method_name = self.log_model.__name__

        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                serialization_format=self.mlflow_save_format,
                registered_model_name=model_name,
                artifact_path=model_name,
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Logged {model_name} in mlflow",
            )

        except Exception as e:
            self.log_writer.self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def log_metric(self, model_name, metric):
        method_name = self.log_metric.__name__

        try:
            mlflow.log_metric(key=model_name + "-best_score", value=metric)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=model_name + "-best score logged in mlflow",
            )

        except Exception as e:
            self.log_writer.self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def log_param(self, model, model_name, param):
        method_name = self.log_param.__name__

        try:
            name = model_name + f"-{param}"

            mlflow.log_param(key=name, value=model.__dict__[param])

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"{name} logged in mlflow",
            )

        except Exception as e:
            self.log_writer.self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def log_all_for_model(self, idx, model, model_param_name, model_score):
        method_name = self.log_all_for_model.__name__

        try:
            base_model_name = get_model_name(
                model=model, db_name=self.db_name, collection_name=self.collection_name
            )

            model_name = base_model_name + str(idx)

            model_params_list = list(
                self.config["model_params"][model_param_name].keys()
            )

            for param in model_params_list:
                self.log_param(model=model, model_name=model_name, param=param)

            self.log_model(model=model, model_name=model_name)

            self.log_metric(model_name=model_name, metric=float(model_score))

        except Exception as e:
            self.log_writer.self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def transition_mlflow_model(
        self, model_version, stage, model_name, bucket, db_name, collection_name
    ):
        method_name = self.transition_mlflow_model.__name__

        try:
            current_version = model_version

            client = self.get_mlflow_client(server_uri=self.remote_server_uri)

            model = model_name + self.model_save_format

            trained_model_file = self.trained_models_dir + "/" + model

            stag_model_file = self.staged_models_dir + "/" + model

            prod_model_file = self.prod_models_dir + "/" + model

            if stage == "Production":
                client.transition_model_version_stage(
                    name=model_name, version=current_version, stage=stage
                )

                self.log_writer.log(
                    db_name=db_name,
                    collection_name=collection_name,
                    log_message="Transitioned "
                    + model_name
                    + "to production in mlflow",
                )

                self.s3_obj.copy_data_to_other_bucket(
                    src_bucket=bucket,
                    src_file=trained_model_file,
                    dest_bucket=bucket,
                    dest_file=prod_model_file,
                    db_name=db_name,
                    collection_name=collection_name,
                )

            elif stage == "Staging":
                client.transition_model_version_stage(
                    name=model_name, version=current_version, stage=stage
                )

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_message="Transitioned " + model_name + "to staging in mlflow",
                )

                self.s3_obj.copy_data_to_other_bucket(
                    src_bucket=bucket,
                    src_file=trained_model_file,
                    dest_bucket=bucket,
                    dest_file=stag_model_file,
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                )

            else:
                self.log_writer.log(
                    db_name=db_name,
                    collection_name=collection_name,
                    log_message="Please select stage for model transition",
                )

        except Exception as e:
            self.log_writer.self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )
