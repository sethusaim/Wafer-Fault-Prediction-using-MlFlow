import os

import pandas as pd
from utils.exception import raise_exception
from utils.logger import App_Logger
from utils.read_params import read_params
from wafer.data_ingestion.data_loader_prediction import Data_Getter_Pred
from wafer.data_preprocessing.preprocessing import Preprocessor
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class prediction:
    """
    Description :   This class shall be used for prediction of new data,based on the models which are
                    in production
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, path):
        self.config = read_params()

        self.s3_obj = S3_Operations()

        self.pred_log = self.config["pred_db_log"]["pred_main"]

        self.db_name = self.config["db_log"]["db_pred_log"]

        self.model_bucket = self.config["s3_bucket"]["wafer_model_bucket"]

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.log_writer = App_Logger()

        self.prod_model_dir = self.config["models_dir"]["prod"]

        self.class_name = self.__class__.__name__

    def prediction_from_model(self):
        """
        Method Name :   prediction_from_model
        Description :   This method is actually responsible for picking the models from the production and
                        predictions on the new data
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        method_name = self.prediction_from_model.__name__

        try:
            self.s3_obj.delete_pred_file(
                db_name=self.db_name, collection_name=self.pred_log
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_log,
                log_message="Start of Prediction",
            )

            data_getter = Data_Getter_Pred(self.db_name, self.pred_log)

            data = data_getter.get_data()

            preprocessor = Preprocessor(self.db_name, self.pred_log)

            is_null_present = preprocessor.is_null_present(data)

            if is_null_present:
                data = preprocessor.impute_missing_values(data)

            cols_to_drop = preprocessor.get_columns_with_zero_std_deviation(data)

            data = preprocessor.remove_columns(data, cols_to_drop)

            kmeans_model_name = os.path.join(self.prod_model_dir, "KMeans")

            kmeans = self.s3_obj.load_model_from_s3(
                bucket=self.model_bucket,
                model_name=kmeans_model_name,
                db_name=self.db_name,
                collection_name=self.pred_log,
            )

            clusters = kmeans.predict(data.drop(["Wafer"], axis=1))

            data["clusters"] = clusters

            clusters = data["clusters"].unique()

            for i in clusters:
                cluster_data = data[data["clusters"] == i]

                wafer_names = list(cluster_data["Wafer"])

                cluster_data = data.drop(labels=["Wafer"], axis=1)

                cluster_data = cluster_data.drop(["clusters"], axis=1)

                model_name = self.s3_obj.find_correct_model_file(
                    cluster_number=i,
                    bucket_name=self.model_bucket,
                    db_name=self.db_name,
                    collection_name=self.pred_log,
                )

                prod_model_name = os.path.join(self.prod_model_dir, model_name)

                model = self.s3_obj.load_model_from_s3(
                    bucket=self.model_bucket,
                    model_name=prod_model_name,
                    db_name=self.db_name,
                    collection_name=self.pred_log,
                )

                result = list(model.predict(cluster_data))

                result = pd.DataFrame(
                    list(zip(wafer_names, result)), columns=["Wafer", "Prediction"]
                )

                path = self.config["pred_output_file"]

                result.to_csv(path, header=True, mode="a+")

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.pred_log,
                    log_message="Local copy of the prediction file is created",
                )

                self.s3_obj.upload_to_s3(
                    src_file=path,
                    bucket=self.input_files_bucket,
                    dest_file=path,
                    db_name=self.db_name,
                    collection_name=self.pred_log,
                )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_log,
                log_message="End of Prediction",
            )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_log,
            )

        return path, result.head().to_json(orient="records")
