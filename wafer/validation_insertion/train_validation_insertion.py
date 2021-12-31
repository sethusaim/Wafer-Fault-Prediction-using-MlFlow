from utils.logger import App_Logger
from utils.read_params import read_params
from wafer.dataTransform.data_transformation_train import dataTransform
from wafer.dataTypeValid.data_type_valid_train import dBOperation
from wafer.raw_data_validation.train_data_validation import Raw_Data_validation


class train_validation:
    """
    Description :   This class shall be used for performing validation on the training data and some perform
                    operations on the raw data
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, raw_data_bucket_name):
        self.raw_data = Raw_Data_validation(raw_data_bucket_name)

        self.dataTransform = dataTransform()

        self.dBOperation = dBOperation()

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.db_name = self.config["db_log"]["db_train_log"]

        self.train_main_log = self.config["train_db_log"]["train_main"]

        self.log_writer = App_Logger()

    def train_validation(self):
        """
        Method Name :   train_validation
        Description :   This method is responsible for validating the training raw data and perform some
                        operations on the traning raw data
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """

        method_name = self.train_validation.__name__

        try:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Start of Validation on files!!",
            )

            (
                LengthOfDateStampInFile,
                LengthOfTimeStampInFile,
                column_names,
                noofcolumns,
            ) = self.raw_data.values_from_schema()

            regex = self.raw_data.get_regex_pattern()

            self.raw_data.create_dirs_for_good_bad_data()

            self.raw_data.validate_raw_file_name(
                regex, LengthOfDateStampInFile, LengthOfTimeStampInFile
            )

            self.raw_data.validate_col_length(noofcolumns)

            self.raw_data.validate_missing_values_in_col()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Raw Data Validation Complete!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Starting Data Transforamtion!!",
            )

            self.dataTransform.rename_target_column()

            self.dataTransform.replace_missing_with_null()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="DataTransformation Completed!!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Insertion of Data into mongodb started!!!!",
            )

            self.dBOperation.insert_good_data_as_record(
                db_name=self.config["mongodb"]["wafer_data_db_name"],
                collection_name=self.config["mongodb"]["wafer_train_data_collection"],
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Insertion in Table completed!!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Extracting csv file from table",
            )

            self.dBOperation.export_collection_to_csv(
                db_name=self.config["mongodb"]["wafer_data_db_name"],
                collection_name=self.config["mongodb"]["wafer_train_data_collection"],
            )

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message=exception_msg,
            )

            raise Exception(exception_msg)
