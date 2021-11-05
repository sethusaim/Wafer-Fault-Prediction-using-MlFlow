import os

import mlflow
from sklearn.model_selection import train_test_split
from src.data_preprocessing.clustering import KMeansClustering
from src.data_preprocessing.data_ingestion.data_loader_train import Data_Getter
from src.data_preprocessing.preprocessing import Preprocessor
from src.file_operations.file_methods import File_Operation
from src.model_finder.tuner import Model_Finder
from utils.application_logging.logger import App_Logger
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

        self.train_model_log = os.path.join(
            self.config["log_dir"]["train_log_dir"], "ModelTrainingLog.txt"
        )

        self.file_object = open(self.train_model_log, "a+")

    def trainingModel(self):
        """
        Method Name :   trainingModel
        Description :   This method is actually responsible for training the selected machine learning models
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        self.log_writer.log(self.file_object, "Start of Training")

        try:
            data_getter = Data_Getter(self.file_object, self.log_writer)

            data = data_getter.get_data()

            preprocessor = Preprocessor(self.file_object, self.log_writer)

            data = preprocessor.remove_columns(data, ["Wafer"])

            X, Y = preprocessor.separate_label_feature(
                data, label_column_name=self.config["base"]["target_col"]
            )

            is_null_present = preprocessor.is_null_present(X)

            if is_null_present:
                X = preprocessor.impute_missing_values(X)

            cols_to_drop = preprocessor.get_columns_with_zero_std_deviation(X)

            X = preprocessor.remove_columns(X, cols_to_drop)

            kmeans = KMeansClustering(self.file_object, self.log_writer)

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

                model_finder = Model_Finder(self.file_object, self.log_writer)

                (
                    rf_model,
                    rf_model_score,
                    xgb_model,
                    xgb_model_score,
                ) = model_finder.get_trained_models(x_train, y_train, x_test, y_test)

                file_op = File_Operation(self.file_object, self.log_writer)

                saved_rf_model = file_op.save_model(
                    rf_model, self.config["model_names"]["rf_model_name"] + str(i)
                )

                self.log_writer.log(
                    self.file_object,
                    "Saved "
                    + self.config["model_names"]["rf_model_name"]
                    + str(i)
                    + " in trained model folder",
                )

                saved_xgb_model = file_op.save_model(
                    xgb_model, self.config["model_names"]["xgb_model_name"] + str(i)
                )

                self.log_writer.log(
                    self.file_object,
                    "Saved "
                    + self.config["model_names"]["xgb_model_name"]
                    + str(i)
                    + " in trained model folder",
                )

                try:
                    self.log_writer.log(
                        self.file_object, "Started setting the remote server uri"
                    )

                    remote_server_uri = self.config["mlflow_config"][
                        "remote_server_uri"
                    ]

                    mlflow.set_tracking_uri(remote_server_uri)

                    self.log_writer.log(
                        self.file_object, "Setting of remote server uri done"
                    )

                    self.log_writer.log(
                        self.file_object,
                        "Started setting the experiment name in mlflow",
                    )
                    try:
                        exp_name = self.config["mlflow_config"]["experiment_name"]

                        self.log_writer.log(self.file_object, "Got the experiment name")

                        s3_bucket = self.config["mlflow_config"]["s3_bucket_config"]

                        self.log_writer.log(
                            self.file_object, "set s3 bucket configuration"
                        )

                        mlflow.create_experiment(
                            name=exp_name, artifact_location=s3_bucket
                        )

                        self.log_writer.log(
                            self.file_object,
                            f"experiment name : {exp_name} has been created",
                        )

                        mlflow.set_experiment(experiment_name=exp_name)

                        self.log_writer.log(
                            self.file_object,
                            f"Experiment name has been set to {exp_name}",
                        )

                    except:
                        mlflow.get_experiment_by_name(
                            self.config["mlflow_config"]["experiment_name"]
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Experiment name already exists,using the previous one",
                        )

                    self.log_writer.log(
                        self.file_object, "Setting of experiment name is done"
                    )

                    self.log_writer.log(
                        self.file_object,
                        "Started mlflow server with "
                        + self.config["mlflow_config"]["run_name"],
                    )

                    with mlflow.start_run(
                        run_name=self.config["mlflow_config"]["run_name"]
                    ):
                        self.log_writer.log(
                            self.file_object,
                            "Started logging of models,metrics and params in mlflow",
                        )

                        self.log_writer.log(
                            self.file_object, "Starting logging of kmeans model"
                        )

                        mlflow.sklearn.log_model(
                            kmeans_model,
                            artifact_path=self.config["mlflow_config"]["artifacts_dir"],
                            serialization_format=self.config["mlflow_config"][
                                "serialization_format"
                            ],
                            registered_model_name=self.config["model_names"][
                                "kmeans_model_name"
                            ],
                        )

                        self.log_writer.log(
                            self.file_object, "Logged kmeans model in mlflow"
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Started logging "
                            + self.config["model_names"]["xgb_model_name"]
                            + str(i)
                            + " learning rate in mlflow",
                        )

                        mlflow.log_param(
                            self.config["model_names"]["xgb_model_name"]
                            + str(i)
                            + "-learning_rate",
                            xgb_model.learning_rate,
                        )

                        self.log_writer.log(
                            self.file_object,
                            self.config["model_names"]["xgb_model_name"]
                            + str(i)
                            + " learning rate logged in mlflow",
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Started logging "
                            + self.config["model_names"]["xgb_model_name"]
                            + str(i)
                            + " max depth in mlflow",
                        )

                        mlflow.log_param(
                            self.config["model_names"]["xgb_model_name"]
                            + str(i)
                            + "-max_depth",
                            xgb_model.max_depth,
                        )

                        self.log_writer.log(
                            self.file_object,
                            self.config["model_names"]["xgb_model_name"]
                            + str(i)
                            + " max depth logged in mlflow",
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Started logging "
                            + self.config["model_names"]["xgb_model_name"]
                            + str(i)
                            + " n_estimators in mlflow",
                        )

                        mlflow.log_param(
                            self.config["model_names"]["xgb_model_name"]
                            + str(i)
                            + "-n_estimators",
                            xgb_model.n_estimators,
                        )

                        self.log_writer.log(
                            self.file_object,
                            self.config["model_names"]["xgb_model_name"]
                            + str(i)
                            + " n_estimators logged in mlflow",
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Started logging "
                            + self.config["model_names"]["rf_model_name"]
                            + str(i)
                            + " criterion in mlflow",
                        )

                        mlflow.log_param(
                            self.config["model_names"]["rf_model_name"]
                            + str(i)
                            + "-criterion",
                            rf_model.criterion,
                        )

                        self.log_writer.log(
                            self.file_object,
                            self.config["model_names"]["rf_model_name"]
                            + str(i)
                            + " criterion logged in mlflow",
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Started logging "
                            + self.config["model_names"]["rf_model_name"]
                            + str(i)
                            + " max depth in mlflow",
                        )

                        mlflow.log_param(
                            self.config["model_names"]["rf_model_name"]
                            + str(i)
                            + "-max_depth",
                            rf_model.max_depth,
                        )

                        self.log_writer.log(
                            self.file_object,
                            self.config["model_names"]["rf_model_name"]
                            + str(i)
                            + "-max_features logged in mlflow",
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Started logging "
                            + self.config["model_names"]["rf_model_name"]
                            + str(i)
                            + "n_estimators",
                        )

                        mlflow.log_param(
                            self.config["model_names"]["rf_model_name"]
                            + str(i)
                            + "-n_estimators",
                            rf_model.n_estimators,
                        )

                        self.log_writer.log(
                            self.file_object,
                            self.config["model_names"]["rf_model_name"]
                            + str(i)
                            + "-n_estimatores logged in mlflow",
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Started logging "
                            + self.config["model_names"]["xgb_model_name"]
                            + str(i)
                            + " best score in mlflow",
                        )

                        mlflow.log_metric(
                            self.config["model_names"]["xgb_model_name"]
                            + str(i)
                            + "-best_score",
                            xgb_model_score,
                        )

                        self.log_writer.log(
                            self.file_object,
                            self.config["model_names"]["xgb_model_name"]
                            + str(i)
                            + " best_score in mlflow",
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Started logging "
                            + self.config["model_names"]["rf_model_name"]
                            + str(i)
                            + " best score in mlflow",
                        )

                        mlflow.log_metric(
                            self.config["model_names"]["rf_model_name"]
                            + str(i)
                            + "-best_score",
                            rf_model_score,
                        )

                        self.log_writer.log(
                            self.file_object,
                            self.config["model_names"]["rf_model_name"]
                            + str(i)
                            + " best_score logged in mlflow",
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Started logging "
                            + self.config["model_names"]["xgb_model_name"]
                            + str(i)
                            + " model in mlflow",
                        )

                        mlflow.sklearn.log_model(
                            xgb_model,
                            artifact_path=self.config["mlflow_config"]["artifacts_dir"],
                            serialization_format=self.config["mlflow_config"][
                                "serialization_format"
                            ],
                            registered_model_name=self.config["model_names"][
                                "xgb_model_name"
                            ]
                            + str(i),
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Logged "
                            + self.config["model_names"]["xgb_model_name"]
                            + str(i)
                            + " in mlflow",
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Started logging "
                            + self.config["model_names"]["rf_model_name"]
                            + str(i)
                            + " model in mlflow",
                        )

                        mlflow.sklearn.log_model(
                            rf_model,
                            artifact_path=self.config["mlflow_config"]["artifacts_dir"],
                            serialization_format=self.config["mlflow_config"][
                                "serialization_format"
                            ],
                            registered_model_name=self.config["model_names"][
                                "rf_model_name"
                            ]
                            + str(i),
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Logged "
                            + self.config["model_names"]["rf_model_name"]
                            + str(i)
                            + " in mlflow",
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Logged params,scores and models for cluster " + str(i),
                        )

                except Exception as e:
                    self.log_writer.log(self.file_object, str(e))

                    self.log_writer.log(
                        self.file_object,
                        "Mlflow logging of params,metrics and models failed",
                    )

                    raise e

            self.log_writer.log(
                self.file_object,
                "Logging of params,scores and models successfull in mlflow",
            )

            self.log_writer.log(self.file_object, "Successful End of Training")

            self.file_object.close()

            return number_of_clusters

        except Exception as e:
            self.log_writer.log(self.file_object, "Error Occurred : " + str(e))

            self.log_writer.log(self.file_object, "Unsuccessful End of Training")

            self.file_object.close()

            raise e
