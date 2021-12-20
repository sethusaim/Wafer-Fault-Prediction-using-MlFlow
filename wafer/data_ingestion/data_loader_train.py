from utils.logger import App_Logger
from utils.main_utils import (
    convert_object_to_bytes,
    get_dataframe_from_bytes,
    read_params,
)
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class Data_Getter:
    """
    Description :   This class shall be used for obtaining the df from the source for training
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, db_name, logger_object):
        self.config = read_params()

        self.training_file = self.config["export_train_csv_file"]

        self.db_name = db_name

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.s3_obj = S3_Operations()

        self.logger_object = logger_object

        self.log_writter = App_Logger()

    def get_data(self):
        """
        Method Name :   get_data
        Description :   This method reads the data from the source
        Output      :   A pandas dataframe
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """

        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.logger_object,
            log_message="Entered the get_data method of the Data_Getter class",
        )

        try:
            csv_obj = self.s3_obj.get_file_object_from_s3(
                bucket=self.input_files_bucket, filename=self.training_file
            )

            csv_data = convert_object_to_bytes(csv_obj)

            df = get_dataframe_from_bytes(csv_data)

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Data Load Successful.Exited the get_data method of the Data_Getter class",
            )

            return df

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message=f"Exception occured in Class : Data_Getter, Method : get_data, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Data Load Unsuccessful.Exited the get_data method of the Data_Getter class",
            )

            raise Exception(
                "Exception occured in Class : Data_Getter, Method : get_data, Error : ",
                str(e),
            )
