from utils.logger import App_Logger
from utils.read_params import read_params
from wafer.s3_bucket_operations.s3_operations import S3_Operation


class Data_Getter_Train:
    """
    Description :   This class shall be used for obtaining the df from the input files s3 bucket where the training file is present
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self, log_file):
        self.config = read_params()

        self.log_file = log_file

        self.train_csv_file = self.config["export_csv_file"]["train"]

        self.input_files_bucket = self.config["s3_bucket"]["input_files"]

        self.s3 = S3_Operation()

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

    def get_data(self):
        """
        Method Name :   get_data
        Description :   This method reads the data from the input files s3 bucket where the training file is stored
        Output      :   A pandas dataframe
        
        On Failure  :   Write an exception log and then raise exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_data.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            log_file=self.log_file,
        )

        try:
            df = self.s3.read_csv(
                fname=self.train_csv_file,
                bucket=self.input_files_bucket,
                log_file=self.log_file,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.log_file,
            )

            return df

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.log_file,
            )
