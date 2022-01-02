from utils.exception import raise_exception
from utils.logger import App_Logger
from utils.main_utils import convert_object_to_dataframe
from utils.read_params import read_params
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class Data_Getter_Pred:
    """
    Description :   This class shall be used for obtaining the data from the source for prediction
    Written By  :   iNeuron Intelligence
    Version     :   1.0
    Revision    :   None
    """

    def __init__(self, db_name, collection_name):
        self.config = read_params()

        self.prediction_file = self.config["export_pred_csv_file"]

        self.db_name = db_name

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.s3_obj = S3_Operations()

        self.collection_name = collection_name

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

    def get_data(self):
        """
        Method Name :   get_data
        Description :   This method reads the data from the source
        Written by  :   iNeuron Intelligence
        Output      :   a pandas dataframe
        On failure  :   Raise Exception
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """

        method_name = self.get_data.__name__

        self.log_writer.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message=f"Entered the {method_name} method of the {self.class_name} class",
        )

        try:
            csv_obj = self.s3_obj.get_file_objects_from_s3(
                bucket=self.input_files_bucket,
                filename=self.prediction_file,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            df = convert_object_to_dataframe(
                obj=csv_obj, db_name=self.db_name, collection_name=self.collection_name
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Data Load Successful.Exited the get_data method of the Data_Getter class",
            )

            return df

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Data Load Unsuccessful.Exited the get_data method of the Data_Getter class",
            )

            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )
