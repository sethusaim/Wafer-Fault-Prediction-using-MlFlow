from utils.logger import App_Logger
from utils.main_utils import read_params
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
            ) = self.raw_data.valuesFromSchema()

            regex = self.raw_data.manualRegexCreation()

            self.raw_data.validationFileNameRaw(
                regex, LengthOfDateStampInFile, LengthOfTimeStampInFile
            )

            self.raw_data.validateColumnLength(noofcolumns)

            self.raw_data.validateMissingValuesInWholeColumn()

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

            self.dataTransform.replaceMissingWithNull()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="DataTransformation Completed!!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Creating Training_Database and tables on the basis of given schema!!!",
            )

            self.dBOperation.createTableDb("Training", column_names)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Table creation Completed!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Insertion of Data into Table started!!!!",
            )

            self.dBOperation.insertIntoTableGoodData("Training")

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Insertion in Table completed!!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Validation Operation completed!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Extracting csv file from table",
            )

            self.dBOperation.selectingDatafromtableintocsv("Training")

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message=f"Exception occured in Class : train_validation, Method : train_validation, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : train_validation, Method : train_validation, Error : ",
                str(e),
            )
