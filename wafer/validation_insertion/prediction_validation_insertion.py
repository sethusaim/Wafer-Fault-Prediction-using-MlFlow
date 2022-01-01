from utils.logger import App_Logger
from utils.read_params import read_params
from wafer.data_transform.data_transformation_pred import data_transformPredict
from wafer.data_type_valid.data_type_valid_pred import dBOperation
from wafer.raw_data_validation.pred_data_validation import \
    Prediction_Data_validation


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

        self.data_transform = data_transformPredict()

        self.dBOperation = dBOperation()

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
            ) = self.raw_data.valuesFromSchema()

            regex = self.raw_data.manualRegexCreation()

            self.raw_data.validationFileNameRaw(
                regex, LengthOfDateStampInFile, LengthOfTimeStampInFile
            )

            self.raw_data.validateColumnLength(noofcolumns)

            self.raw_data.validateMissingValuesInWholeColumn()

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

            self.data_transform.replaceMissingWithNull()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="data_transformation Completed!!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Creating Prediction_Database and tables on the basis of given schema!!!",
            )

            self.dBOperation.createTableDb("Prediction", column_names)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Table creation Completed!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Insertion of Data into Table started!!!!",
            )

            self.dBOperation.insertIntoTableGoodData("Prediction")

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Insertion in Table completed!!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Deleting Good Data Folder!!!",
            )

            self.raw_data.deleteExistingGoodDataTrainingFolder()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Good_Data folder deleted!!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Moving bad files to Archive and deleting Bad_Data folder!!!",
            )

            self.raw_data.moveBadFilesToArchiveBad()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Bad files moved to archive!! Bad folder Deleted!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Validation Operation completed!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Extracting csv file from table",
            )

            self.dBOperation.selectingDatafromtableintocsv("Prediction")

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message=f"Exception occured in Class : pred_validation. \
                    Method : prediction_validation, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : pred_validation. \
                    Method : prediction_validation, Error : ",
                str(e),
            )
