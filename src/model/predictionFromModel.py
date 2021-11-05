import os

import pandas as pd
from src.data_preprocessing.data_ingestion.data_loader_prediction import (
    Data_Getter_Pred,
)
from src.data_preprocessing.preprocessing import Preprocessor
from src.file_operations import file_methods
from src.raw_data_validation.pred_data_validation import Prediction_Data_validation
from utils.application_logging.logger import App_Logger
from utils.read_params import read_params


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

        self.pred_log = os.path.join(
            self.config["log_dir"]["pred_log_dir"], "Prediction_Log.txt"
        )

        self.file_object = open(self.pred_log, "a+")

        self.log_writer = App_Logger()

        if path is not None:
            self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):
        """
        Method Name :   predictionFromModel
        Description :   This method is actually responsible for picking the models from the production and
                        predictions on the new data
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        try:
            self.pred_data_val.deletePredictionFile()

            self.log_writer.log(self.file_object, "Start of Prediction")

            data_getter = Data_Getter_Pred(self.file_object, self.log_writer)

            data = data_getter.get_data()

            preprocessor = Preprocessor(self.file_object, self.log_writer)

            is_null_present = preprocessor.is_null_present(data)

            if is_null_present:
                data = preprocessor.impute_missing_values(data)

            cols_to_drop = preprocessor.get_columns_with_zero_std_deviation(data)

            data = preprocessor.remove_columns(data, cols_to_drop)

            file_loader = file_methods.File_Operation(self.file_object, self.log_writer)

            kmeans = file_loader.load_model(
                self.config["model_names"]["kmeans_model_name"]
            )

            clusters = kmeans.predict(data.drop(["Wafer"], axis=1))

            data["clusters"] = clusters

            clusters = data["clusters"].unique()

            for i in clusters:
                cluster_data = data[data["clusters"] == i]

                wafer_names = list(cluster_data["Wafer"])

                cluster_data = data.drop(labels=["Wafer"], axis=1)

                cluster_data = cluster_data.drop(["clusters"], axis=1)

                model_name = file_loader.find_correct_model_file(i)

                model = file_loader.load_model(model_name)

                result = list(model.predict(cluster_data))

                result = pd.DataFrame(
                    list(zip(wafer_names, result)), columns=["Wafer", "Prediction"]
                )

                path = self.config["pred_output_file"]

                result.to_csv(
                    self.config["pred_output_file"],
                    header=True,
                    mode="a+",
                )

            self.log_writer.log(self.file_object, "End of Prediction")

        except Exception as ex:
            self.log_writer.log(
                self.file_object,
                "Error occured while running the prediction!! Error:: %s" % ex,
            )

            raise ex

        return path, result.head().to_json(orient="records")
