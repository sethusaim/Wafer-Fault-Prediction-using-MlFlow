from utils.exception import raise_exception_log
from utils.logger import App_Logger
from utils.main_utils import convert_object_to_dataframe
from utils.read_params import read_params
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class Data_Getter:
    """
    Description :   This class shall be used for obtaining the df from the source for training
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, db_name, collection_name):
        self.config = read_params()

        self.training_file = self.config["export_train_csv_file"]

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
        Output      :   A pandas dataframe
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
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
                filename=self.training_file,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            df = convert_object_to_dataframe(
                obj=csv_obj, db_name=self.db_name, collection_name=self.collection_name
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Data Load Successful.Exited the {method_name} method of the {self.class_name} class",
            )

            return df

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Data Load Unsuccessful.Exited the {method_name} method of the {self.class_name} class",
            )

            raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )
