import os

from src.dataTransform.data_transformation_pred import dataTransformPredict
from src.dataTypeValid.data_type_valid_pred import dBOperation
from src.raw_data_validation.pred_data_validation import Prediction_Data_validation
from utils.application_logging.logger import App_Logger
from utils.main_utils import read_params


class pred_validation:
    """
    Description :   This class shall be used for validating all the prediction raw data and then perform
                    some operations on the data
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, path):
        self.raw_data = Prediction_Data_validation(path)

        self.dataTransform = dataTransformPredict()

        self.dBOperation = dBOperation()

        self.config = read_params()

        self.pred_log = os.path.join(
            self.config["log_dir"]["pred_log_dir"], "Prediction_Log.txt"
        )

        self.file_object = open(self.pred_log, "a+")

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
        try:
            self.log_writer.log(
                self.file_object, "Start of Validation on files for prediction!!"
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

            self.log_writer.log(self.file_object, "Raw Data Validation Complete!!")

            self.log_writer.log(self.file_object, "Starting Data Transforamtion!!")

            self.dataTransform.replaceMissingWithNull()

            self.log_writer.log(self.file_object, "DataTransformation Completed!!!")

            self.log_writer.log(
                self.file_object,
                "Creating Prediction_Database and tables on the basis of given schema!!!",
            )

            self.dBOperation.createTableDb("Prediction", column_names)

            self.log_writer.log(self.file_object, "Table creation Completed!!")

            self.log_writer.log(
                self.file_object, "Insertion of Data into Table started!!!!"
            )

            self.dBOperation.insertIntoTableGoodData("Prediction")

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

            self.dBOperation.selectingDatafromtableintocsv("Prediction")

        except Exception as e:
            self.log_writer.log(self.file_object, "Error Occurred : " + str(e))

            raise e
