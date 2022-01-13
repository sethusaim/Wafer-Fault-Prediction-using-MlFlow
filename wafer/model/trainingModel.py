import mlflow
from sklearn.model_selection import train_test_split

from utils.logger import App_Logger
from utils.main_utils import get_model_name
from utils.read_params import read_params
from wafer.data_ingestion.data_loader_train import Data_Getter
from wafer.data_preprocessing.clustering import KMeansClustering
from wafer.data_preprocessing.preprocessing import Preprocessor
from wafer.mlflow_utils.mlflow_operations import Mlflow_Operations
from wafer.model_finder.tuner import Model_Finder
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class train_model:
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

        self.target_col = self.config["base"]["target_col"]

        self.test_size = self.config["base"]["test_size"]

        self.random_state = self.config["base"]["random_state"]

        self.model_bucket = self.config["s3_bucket"]["wafer_model_bucket"]

        self.remote_server_uri = self.config["mlflow_config"]["remote_server_uri"]

        self.experiment_name = self.config["mlflow_config"]["experiment_name"]

        self.run_name = self.config["mlflow_config"]["run_name"]

        self.class_name = self.__class__.__name__

        self.data_getter = Data_Getter(
            db_name=self.db_name, collection_name=self.model_train_log
        )

        self.preprocessor = Preprocessor(
            db_name=self.db_name, collection_name=self.model_train_log
        )

        self.kmeans = KMeansClustering(
            db_name=self.db_name, collection_name=self.model_train_log
        )

        self.model_finder = Model_Finder(self.db_name, self.model_train_log)

        self.mlflow_op = Mlflow_Operations(
            db_name=self.db_name, collection_name=self.model_train_log
        )

        self.s3_obj = S3_Operations()

    def training_model(self):
        """
        Method Name :   trainingModel
        Description :   This method is actually responsible for training the selected machine learning models
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """

        method_name = self.training_model.__name__

        self.log_writer.log(
            db_name=self.db_name,
            collection_name=self.model_train_log,
            log_message="Start of Training",
        )

        try:
            data = self.data_getter.get_data()

            data = self.preprocessor.remove_columns(data, ["Wafer"])

            X, Y = self.preprocessor.separate_label_feature(
                data, label_column_name=self.target_col
            )

            is_null_present = self.preprocessor.is_null_present(X)

            if is_null_present:
                X = self.preprocessor.impute_missing_values(X)

            cols_to_drop = self.preprocessor.get_columns_with_zero_std_deviation(X)

            X = self.preprocessor.remove_columns(X, cols_to_drop)

            number_of_clusters = self.kmeans.elbow_plot(X)

            X, kmeans_model = self.kmeans.create_clusters(X, number_of_clusters)

            X["Labels"] = Y

            list_of_clusters = X["Cluster"].unique()

            self.s3_obj.create_folders_for_prod_and_stag(
                bucket_name=self.model_bucket,
                db_name=self.db_name,
                collection_name=self.model_train_log,
            )

            for i in list_of_clusters:
                cluster_data = X[X["Cluster"] == i]

                cluster_features = cluster_data.drop(["Labels", "Cluster"], axis=1)

                cluster_label = cluster_data["Labels"]

                x_train, x_test, y_train, y_test = train_test_split(
                    cluster_features,
                    cluster_label,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )

                (
                    rf_model,
                    rf_model_score,
                    xgb_model,
                    xgb_model_score,
                ) = self.model_finder.get_trained_models(
                    x_train, y_train, x_test, y_test
                )

                kmeans_model_name = get_model_name(
                    model=kmeans_model,
                    db_name=self.db_name,
                    collection_name=self.model_train_log,
                )

                self.s3_obj.save_model_to_s3(
                    idx=i,
                    model=rf_model,
                    model_bucket=self.model_bucket,
                    db_name=self.db_name,
                    collection_name=self.model_train_log,
                )

                self.s3_obj.save_model_to_s3(
                    idx=i,
                    model=xgb_model,
                    model_bucket=xgb_model,
                    db_name=self.db_name,
                    collection_name=self.model_train_log,
                )

                try:
                    self.mlflow_op.set_mlflow_tracking_uri(
                        server_uri=self.remote_server_uri
                    )

                    self.mlflow_op.set_mlflow_experiment(
                        experiment_name=self.experiment_name
                    )

                    with mlflow.start_run(run_name=self.run_name):
                        self.mlflow_op.log_model(
                            model=kmeans_model, model_name=kmeans_model_name
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
                        log_message="Mlflow logging of params,metrics and models failed",
                    )

                    self.log_writer.raise_exception_log(
                        error=e,
                        class_name=self.class_name,
                        method_name=method_name,
                        db_name=self.db_name,
                        collection_name=self.model_train_log,
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
                log_message="Unsuccessful End of Training",
            )

            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.model_train_log,
            )
