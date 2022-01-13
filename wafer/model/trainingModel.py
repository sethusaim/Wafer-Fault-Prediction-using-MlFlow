import mlflow
from sklearn.model_selection import train_test_split
from utils.logger import App_Logger
from utils.model_utils import get_model_name
from utils.read_params import read_params
from wafer.data_ingestion.data_loader_train import data_getter_train
from wafer.data_preprocessing.clustering import KMeansClustering
from wafer.data_preprocessing.preprocessing import Preprocessor
from wafer.mlflow_utils.mlflow_operations import Mlflow_Operations
from wafer.model_finder.tuner import Model_Finder
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class train_model:
    """
    Description :   This method is used for getting the data and applying 
                    some preprocessing steps and then train the models and register them in mlflow

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.db_name = self.config["db_log"]["db_train_log"]

        self.model_train_log = self.config["train_db_log"]["model_training"]

        self.model_bucket = self.config["s3_bucket"]["wafer_model_bucket"]

        self.test_size = self.config["base"]["test_size"]

        self.target_col = self.config["base"]["target_col"]

        self.random_state = self.config["base"]["random_state"]

        self.remote_server_uri = self.config["mlflow_config"]["remote_server_uri"]

        self.experiment_name = self.config["mlflow_config"]["experiment_name"]

        self.run_name = self.config["mlflow_config"]["run_name"]

        self.class_name = self.__class__.__name__

        self.mlflow_op = Mlflow_Operations(
            db_name=self.db_name, collection_name=self.model_train_log
        )

        self.data_getter_train_obj = data_getter_train(
            db_name=self.db_name, collection_name=self.model_train_log
        )

        self.preprocessor_obj = Preprocessor(
            db_name=self.db_name, collection_name=self.model_train_log
        )

        self.kmeans_obj = KMeansClustering(
            db_name=self.db_name, collection_name=self.model_train_log
        )

        self.model_finder_obj = Model_Finder(
            db_name=self.db_name, collection_name=self.model_train_log
        )

        self.s3_obj = S3_Operations()

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
            db_name=self.db_name,
            collection_name=self.model_train_log,
        )

        try:
            df = self.data_getter_train_obj.get_data()

            df = self.preprocessor_obj.replace_invalid_values(data=df)

            df = self.preprocessor_obj.encode_target_cols(data=df)

            is_null_present = self.preprocessor_obj.is_null_present(data=df)

            if is_null_present:
                df = self.preprocessor_obj.impute_missing_values(data=df)

            X, Y = self.preprocessor_obj.separate_label_feature(
                data=df, label_column_name=self.target_col
            )

            cols_to_drop = self.preprocessor_obj.get_columns_with_zero_std_deviation(
                data=X
            )

            X = self.preprocessor_obj.remove_columns(data=X, columns=cols_to_drop)

            X = self.preprocessor_obj.scale_numerical_columns(data=X)

            X = self.preprocessor_obj.apply_pca_transform(X_scaled_data=X)

            X, Y = self.preprocessor_obj.apply_smote(X=X, Y=Y)

            number_of_clusters = self.kmeans_obj.elbow_plot(data=X)

            X, kmeans_model = self.kmeans_obj.create_clusters(
                data=X, number_of_clusters=number_of_clusters
            )

            X["Labels"] = Y

            list_of_clusters = X["Cluster"].unique()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.model_train_log,
                log_message="Got unique list of clusters",
            )

            for i in list_of_clusters:
                cluster_data = X[X["Cluster"] == i]

                cluster_features = cluster_data.drop(["Labels", "Cluster"], axis=1)

                cluster_label = cluster_data["Labels"]

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.model_train_log,
                    log_message="Seprated cluster features and cluster label for the cluster data",
                )

                x_train, x_test, y_train, y_test = train_test_split(
                    cluster_features,
                    cluster_label,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.model_train_log,
                    log_message=f"Performed train test split with test size as {self.test_size} and random state as {self.random_state}",
                )

                (
                    knn_model,
                    knn_model_score,
                    svm_model,
                    svm_model_score,
                ) = self.model_finder_obj.get_trained_models(
                    x_train, y_train, x_test, y_test
                )

                kmeans_model_name = get_model_name(
                    model=kmeans_model,
                    db_name=self.db_name,
                    collection_name=self.model_train_log,
                )

                self.s3_obj.save_model_to_s3(
                    idx=i,
                    model=knn_model,
                    model_bucket=self.model_bucket,
                    db_name=self.db_name,
                    collection_name=self.model_train_log,
                )

                self.s3_obj.save_model_to_s3(
                    idx=i,
                    model=svm_model,
                    model_bucket=self.model_bucket,
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
                            model=knn_model,
                            model_param_name="knn_model",
                            model_score=knn_model_score,
                        )

                        self.mlflow_op.log_all_for_model(
                            idx=i,
                            model=svm_model,
                            model_param_name="svm_model",
                            model_score=svm_model_score,
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
