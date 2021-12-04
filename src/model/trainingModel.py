import mlflow
from sklearn.model_selection import train_test_split
from src.data_preprocessing.clustering import KMeansClustering
from src.data_preprocessing.data_ingestion.data_loader_train import Data_Getter
from src.data_preprocessing.preprocessing import Preprocessor
from src.file_operations.file_methods import File_Operation
from src.model_finder.tuner import Model_Finder
from utils.logger import App_Logger
from utils.main_utils import (
    log_metric_to_mlflow,
    log_model_to_mlflow,
    log_param_to_mlflow,
)
from utils.read_params import read_params


class trainModel:
    """
    Description :   This is the entry point for training the machine learning model
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.db_name = self.config["db_log"]["db_train_log"]

        self.model_train_log = self.config["train_db_log"]["model_training"]

    def trainingModel(self):
        """
        Method Name :   trainingModel
        Description :   This method is actually responsible for training the selected machine learning models
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """

        self.log_writer.log(
            db_name=self.db_name,
            collection_name=self.model_train_log,
            log_message="Start of Training",
        )

        try:
            data_getter = Data_Getter(self.db_name, self.model_train_log)

            data = data_getter.get_data()

            preprocessor = Preprocessor(self.db_name, self.model_train_log)

            data = preprocessor.remove_columns(data, ["Wafer"])

            X, Y = preprocessor.separate_label_feature(
                data, label_column_name=self.config["base"]["target_col"]
            )

            is_null_present = preprocessor.is_null_present(X)

            if is_null_present:
                X = preprocessor.impute_missing_values(X)

            cols_to_drop = preprocessor.get_columns_with_zero_std_deviation(X)

            X = preprocessor.remove_columns(X, cols_to_drop)

            kmeans = KMeansClustering(self.db_name, self.model_train_log)

            number_of_clusters = kmeans.elbow_plot(X)

            X, kmeans_model = kmeans.create_clusters(X, number_of_clusters)

            X["Labels"] = Y

            list_of_clusters = X["Cluster"].unique()

            for i in list_of_clusters:
                cluster_data = X[X["Cluster"] == i]

                cluster_features = cluster_data.drop(["Labels", "Cluster"], axis=1)

                cluster_label = cluster_data["Labels"]

                x_train, x_test, y_train, y_test = train_test_split(
                    cluster_features,
                    cluster_label,
                    test_size=self.config["base"]["test_size"],
                    random_state=self.config["base"]["random_state"],
                )

                model_finder = Model_Finder(self.db_name, self.model_train_log)

                (
                    rf_model,
                    rf_model_score,
                    xgb_model,
                    xgb_model_score,
                ) = model_finder.get_trained_models(x_train, y_train, x_test, y_test)

                file_op = File_Operation(self.db_name, self.model_train_log)

                saved_rf_model = file_op.save_model(
                    rf_model, self.config["model_names"]["rf_model_name"] + str(i)
                )

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.model_train_log,
                    log_message="Saved "
                    + self.config["model_names"]["rf_model_name"]
                    + str(i)
                    + " in trained model folder",
                )

                saved_xgb_model = file_op.save_model(
                    xgb_model, self.config["model_names"]["xgb_model_name"] + str(i)
                )

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.model_train_log,
                    log_message="Saved "
                    + self.config["model_names"]["xgb_model_name"]
                    + str(i)
                    + " in trained model folder",
                )

                try:
                    remote_server_uri = self.config["mlflow_config"][
                        "remote_server_uri"
                    ]

                    mlflow.set_tracking_uri(remote_server_uri)

                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.model_train_log,
                        log_message="Set the remote server uri",
                    )

                    s3_bucket = self.config["mlflow_config"]["s3_bucket"]

                    exp_name = self.config["mlflow_config"]["experiment_name"]

                    try:
                        mlflow.create_experiment(
                            name=exp_name, artifact_location=s3_bucket
                        )

                        self.log_writer.log(
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                            log_message=f"Created experiment with name {self.config['mlflow_config']['experiment_name']}",
                        )

                    except:
                        mlflow.get_experiment_by_name(name=exp_name)

                        self.log_writer.log(
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                            log_message=f"Got the experiment name {self.config['mlflow_config']['experiment_name']}",
                        )

                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.model_train_log,
                        log_message="Started mlflow server with "
                        + self.config["mlflow_config"]["run_name"],
                    )

                    with mlflow.start_run(
                        run_name=self.config["mlflow_config"]["run_name"]
                    ):
                        log_model_to_mlflow(
                            model=kmeans_model,
                            model_name=self.config["model_names"]["kmeans_model_name"],
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_param_to_mlflow(
                            model=xgb_model,
                            model_name=self.config["model_names"]["rf_model_name"]
                            + str(i),
                            param_name="learning_rate",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_param_to_mlflow(
                            model=xgb_model,
                            model_name=self.config["model_names"]["rf_model_name"]
                            + str(i),
                            param_name="max_depth",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_param_to_mlflow(
                            model=xgb_model,
                            model_name=self.config["model_names"]["rf_model_name"]
                            + str(i),
                            param_name="n_estimators",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_param_to_mlflow(
                            model=rf_model,
                            model_name=self.config["model_names"]["rf_model_name"]
                            + str(i),
                            param_name="criterion",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_param_to_mlflow(
                            model=rf_model,
                            model_name=self.config["model_names"]["rf_model_name"]
                            + str(i),
                            param_name="max_features",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_param_to_mlflow(
                            model=rf_model,
                            model_name=self.config["model_names"]["rf_model_name"]
                            + str(i),
                            param_name="n_estimators",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_metric_to_mlflow(
                            model_name=self.config["model_names"]["rf_model_name"]
                            + str(i),
                            metric=xgb_model_score,
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_metric_to_mlflow(
                            model_name=self.config["model_names"]["rf_model_name"]
                            + str(i),
                            metric=rf_model_score,
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_model_to_mlflow(
                            model=xgb_model,
                            model_name=self.config["model_names"]["xgb_model_name"]
                            + str(i),
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_model_to_mlflow(
                            model=rf_model,
                            model_name=self.config["model_names"]["rf_model_name"]
                            + str(i),
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        self.log_writer.log(
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                            log_message="Logged params,scores and models for cluster "
                            + str(i),
                        )

                except Exception as e:
                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.model_train_log,
                        log_message=f"Exception Occured in Class : trainModel, Method : mlflow , Error : {str(e)}",
                    )

                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.model_train_log,
                        log_message="Mlflow logging of params,metrics and models failed",
                    )

                    raise Exception(
                        "Exception Occured in Class : trainModel, Method : mlflow , Error : ",
                        str(e),
                    )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.model_train_log,
                log_message="Logging of params,scores and models successfull in mlflow",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.model_train_log,
                log_message="Successful End of Training",
            )

            return number_of_clusters

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.model_train_log,
                log_message=f"Exception occured in Class : trainModel ,Method : trainingModel, Error : {str(e)}",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.model_train_log,
                log_message="Unsuccessful End of Training",
            )

            raise Exception(
                "Exception occured in Class : trainModel, Method : trainingModel, Error : ",
                str(e),
            )
