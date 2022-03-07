import mlflow
from sklearn.model_selection import train_test_split
from utils.logger import App_Logger
from utils.read_params import read_params
from wafer.data_ingestion.data_loader_train import Data_Getter_Train
from wafer.data_preprocessing.clustering import KMeans_Clustering
from wafer.data_preprocessing.preprocessing import Preprocessor
from wafer.mlflow_utils.mlflow_operations import MLFlow_Operation
from wafer.model_finder.tuner import Model_Finder
from wafer.s3_bucket_operations.s3_operations import S3_Operation


class Train_Model:
    """
    Description :   This method is used for getting the data and applying
                    some preprocessing steps and then train the models and register them in mlflow

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.model_train_log = self.config["train_db_log"]["train_model"]

        self.model_container = self.config["container"]["wafer_model"]

        self.test_size = self.config["base"]["test_size"]

        self.target_col = self.config["base"]["target_col"]

        self.random_state = self.config["base"]["random_state"]

        self.remote_server_uri = self.config["mlflow_config"]["remote_server_uri"]

        self.experiment_name = self.config["mlflow_config"]["experiment_name"]

        self.run_name = self.config["mlflow_config"]["run_name"]

        self.train_model_dir = self.config["models_dir"]["trained"]

        self.class_name = self.__class__.__name__

        self.mlflow_op = MLFlow_Operation(table_name=self.model_train_log)

        self.data_getter_train = Data_Getter_Train(table_name=self.model_train_log)

        self.preprocessor = Preprocessor(table_name=self.model_train_log)

        self.kmeans_op = KMeans_Clustering(table_name=self.model_train_log)

        self.model_finder = Model_Finder(table_name=self.model_train_log)

        self.s3 = S3_Operation()

    def training_model(self):
        """
        Method Name :   training_model
        Description :   This method is used for getting the data and applying
                        some preprocessing steps and then train the models and register them in mlflow

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.training_model.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=self.model_train_log,
        )

        try:
            data = self.data_getter_train.get_data()

            data = self.preprocessor.remove_columns(data, ["Wafer"])

            X, Y = self.preprocessor.separate_label_feature(
                data, label_column_name=self.target_col
            )

            is_null_present = self.preprocessor.is_null_present(X)

            if is_null_present:
                X = self.preprocessor.impute_missing_values(X)

            cols_to_drop = self.preprocessor.get_columns_with_zero_std_deviation(X)

            X = self.preprocessor.remove_columns(X, cols_to_drop)

            number_of_clusters = self.kmeans_op.elbow_plot(X)

            X, kmeans_model = self.kmeans_op.create_clusters(
                data=X, number_of_clusters=number_of_clusters
            )

            X["Labels"] = Y

            list_of_clusters = X["Cluster"].unique()

            for i in list_of_clusters:
                cluster_data = X[X["Cluster"] == i]

                cluster_features = cluster_data.drop(["Labels", "Cluster"], axis=1)

                cluster_label = cluster_data["Labels"]

                self.log_writer.log(
                    table_name=self.model_train_log,
                    log_info="Seprated cluster features and cluster label for the cluster data",
                )

                x_train, x_test, y_train, y_test = train_test_split(
                    cluster_features,
                    cluster_label,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )

                self.log_writer.log(
                    table_name=self.model_train_log,
                    log_info=f"Performed train test split with test size as {self.test_size} and random state as {self.random_state}",
                )

                (
                    xgb_model,
                    xgb_model_score,
                    rf_model,
                    rf_model_score,
                ) = self.model_finder.get_trained_models(
                    x_train, y_train, x_test, y_test
                )

                self.s3.save_model(
                    model=xgb_model,
                    model_dir=self.train_model_dir,
                    container_name=self.model_container,
                    idx=i,
                    table_name=self.model_train_log,
                )

                self.s3.save_model(
                    model=rf_model,
                    idx=i,
                    model_dir=self.train_model_dir,
                    model_container=self.model_container,
                    table_name=self.model_train_log,
                )

                try:
                    self.mlflow_op.set_mlflow_tracking_uri(
                        server_uri=self.remote_server_uri
                    )

                    self.mlflow_op.set_mlflow_experiment(
                        experiment_name=self.experiment_name
                    )

                    with mlflow.start_run(run_name=self.run_name):
                        self.mlflow_op.log_all_for_model(
                            idx=None,
                            model=kmeans_model,
                            model_param_name=None,
                            model_score=None,
                        )

                        self.mlflow_op.log_all_for_model(
                            idx=i,
                            model=xgb_model,
                            model_param_name="xgb_model",
                            model_score=xgb_model_score,
                        )

                        self.mlflow_op.log_all_for_model(
                            idx=i,
                            model=rf_model,
                            model_param_name="rf_model",
                            model_score=rf_model_score,
                        )

                except Exception as e:
                    self.log_writer.log(
                        table_name=self.model_train_log,
                        log_info="Mlflow logging of params,metrics and models failed",
                    )

                    self.log_writer.exception_log(
                        error=e,
                        class_name=self.class_name,
                        method_name=method_name,
                        table_name=self.model_train_log,
                    )

            self.log_writer.log(
                table_name=self.model_train_log, log_info="Successful End of Training",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.model_train_log,
            )

            return number_of_clusters

        except Exception as e:
            self.log_writer.log(
                table_name=self.model_train_log,
                log_info="Unsuccessful End of Training",
            )

            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.model_train_log,
            )
