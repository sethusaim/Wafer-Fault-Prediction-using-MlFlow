from utils.exception import raise_exception
from utils.logger import App_Logger
from utils.read_params import read_params
from wafer.data_transform.data_transformation_pred import data_transform_pred
from wafer.data_type_valid.data_type_valid_pred import dBOperation
from wafer.raw_data_validation.pred_data_validation import Prediction_Data_validation


class pred_validation:
    """
    Description :   This class shall be used for validating all the prediction raw data and then perform
                    some operations on the data
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, bucket_name):
        self.raw_data = Prediction_Data_validation(bucket_name)

        self.data_transform = data_transform_pred()

        self.dBOperation = dBOperation()

        self.class_name = self.__class__.__name__

        self.config = read_params()

        self.db_name = self.config["db_log"]["db_pred_log"]

        self.pred_main_log = self.config["pred_db_log"]["pred_main"]

        self.log_writer = App_Logger()

    def prediction_validation(self):
        """
        Method Name :   prediction_validation
        Description :   This method is responsible for validating the prediction data,using the preprocesssing
                        functions
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        method_name = self.prediction_validation.__name__

        try:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Start of Validation on files for prediction!!",
            )

            (
                LengthOfDateStampInFile,
                LengthOfTimeStampInFile,
                column_names,
                noofcolumns,
            ) = self.raw_data.values_from_schema()

            regex = self.raw_data.get_regex_pattern()

            self.raw_data.validate_raw_file_name(
                regex, LengthOfDateStampInFile, LengthOfTimeStampInFile
            )

            self.raw_data.validate_col_length(noofcolumns)

            self.raw_data.validate_missing_values_in_col()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Raw Data Validation Complete!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Starting Data Transforamtion!!",
            )

            self.data_transform.replace_missing_with_null()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="data_transformation Completed!!!",
            )

            self.dBOperation.insert_good_data_as_record(
                db_name=self.config["mongodb"]["wafer_data_db_name"],
                collection_name=self.config["mongodb"]["wafer_pred_data_collection"],
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Insertion in good data in MongoDB !!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Extracting csv file from MongoDB",
            )

            self.dBOperation.export_collection_to_csv(
                db_name=self.config["mongodb"]["wafer_data_db_name"],
                collection_name=self.config["mongodb"]["wafer_pred_data_collection"],
            )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_main_log,
            )
