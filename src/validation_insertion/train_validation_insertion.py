import os

from src.dataTransform.data_transformation_train import dataTransform
from src.dataTypeValid.data_type_valid_train import dBOperation
from src.raw_data_validation.train_data_validation import Raw_Data_validation
from utils.application_logging.logger import App_Logger
from utils.main_utils import read_params


class train_validation:
    """
    Description :   This class shall be used for performing validation on the training data and some perform
                    operations on the raw data
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, path):
        self.raw_data = Raw_Data_validation(path)

        self.dataTransform = dataTransform()

        self.dBOperation = dBOperation()

        self.config = read_params()

        self.train_main_log = os.path.join(
            self.config["log_dir"]["train_log_dir"], "Training_Main_Log.txt"
        )

        self.file_object = open(self.train_main_log, "a+")

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
            self.log_writer.log(self.file_object, "Start of Validation on files!!")

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

            self.log_writer.log(self.file_object, "Raw Data Validation Complete!!")

            self.log_writer.log(self.file_object, "Starting Data Transforamtion!!")

            self.dataTransform.replaceMissingWithNull()

            self.log_writer.log(self.file_object, "DataTransformation Completed!!!")

            self.log_writer.log(
                self.file_object,
                "Creating Training_Database and tables on the basis of given schema!!!",
            )

            self.dBOperation.createTableDb("Training", column_names)

            self.log_writer.log(self.file_object, "Table creation Completed!!")

            self.log_writer.log(
                self.file_object, "Insertion of Data into Table started!!!!"
            )

            self.dBOperation.insertIntoTableGoodData("Training")

            self.log_writer.log(self.file_object, "Insertion in Table completed!!!")

            self.log_writer.log(self.file_object, "Deleting Good Data Folder!!!")

            self.raw_data.deleteExistingGoodDataTrainingFolder()

            self.log_writer.log(self.file_object, "Good_Data folder deleted!!!")

            self.log_writer.log(
                self.file_object,
                "Moving bad files to Archive and deleting Bad_Data folder!!!",
            )

            self.raw_data.moveBadFilesToArchiveBad()

            self.log_writer.log(
                self.file_object, "Bad files moved to archive!! Bad folder Deleted!!"
            )

            self.log_writer.log(self.file_object, "Validation Operation completed!!")

            self.log_writer.log(self.file_object, "Extracting csv file from table")

            self.dBOperation.selectingDatafromtableintocsv("Training")

            self.file_object.close()

        except Exception as e:
            self.log_writer.log(self.file_object, "Error occurred : " + str(e))

            self.log_writer.log(
                self.file_object, "Error occurred in train_validation class"
            )

            raise e
