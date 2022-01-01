from utils.logger import App_Logger
from utils.main_utils import convert_object_to_dataframe
from utils.read_params import read_params
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class data_transform:
    """
    Description :   This class shall be used for transforming the Good Raw Training data before loaded
                    it in database
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self):
        self.config = read_params()

        self.train_data_bucket = self.config["s3_bucket"]["wafer_train_data_bucket"]

        self.s3_obj = S3_Operations()

        self.log_writer = App_Logger()

        self.good_train_data_dir = self.config["data"]["train"]["good_data_dir"]

        self.class_name = self.__class__.__name__

        self.db_name = self.config["db_log"]["db_train_log"]

        self.train_data_transform_log = self.config["train_db_log"]["data_transform"]

    def rename_target_column(self):
        """
        Method Name :   rename_target_column
        Description :   This method replaces the missing values in columns with "NULL" to store in the table.
                        We are using substring in the first column to keep only "Integer" data for ease up the
                        loading.This columns is anyways going to be removed during training
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """

        method_name = self.rename_target_column.__name__

        try:
            csv_file_objs = self.s3_obj.get_file_objects_from_s3(
                bucket=self.train_data_bucket,
                filename=self.good_train_data_dir,
                db_name=self.db_name,
                collection_name=self.train_data_transform_log,
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_data_transform_log,
                log_message="File transformation started",
            )

            for f in csv_file_objs:
                file = f.key

                abs_f = file.split("/")[-1]

                if file.endswith(".csv"):
                    df = convert_object_to_dataframe(
                        f,
                        db_name=self.db_name,
                        collection_name=self.train_data_transform_log,
                    )

                    df.rename(columns={"Good/Bad": "Output"}, inplace=True)

                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.train_data_transform_log,
                        log_message=f"Renamed the output column for the file for the file {file} ",
                    )

                    df.to_csv(abs_f, index=None, header=True)

                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.train_data_transform_log,
                        log_message=f"Converted {file} to df and local copy copy is created",
                    )

                    self.s3_obj.upload_to_s3(
                        src_file=abs_f,
                        bucket=self.train_data_bucket,
                        dest_file=file,
                        db_name=self.db_name,
                        collection_name=self.train_data_transform_log,
                    )

                else:
                    pass

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_data_transform_log,
                log_message=exception_msg,
            )

            raise Exception(exception_msg)

    def replace_missing_with_null(self):
        """
        Method Name :   replace_missing_with_null
        Description :   This method replaces the missing values in columns with "NULL" to store in the table.
                        We are using substring in the first column to keep only "Integer" data for ease up the
                        loading.This columns is anyways going to be removed during training
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        method_name = self.replace_missing_with_null.__name__

        try:
            csv_file_objs = self.s3_obj.get_file_objects_from_s3(
                bucket=self.train_data_bucket,
                filename=self.good_train_data_dir,
                db_name=self.db_name,
                collection_name=self.train_data_transform_log,
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_data_transform_log,
                log_message="File transformation started",
            )

            for f in csv_file_objs:
                file = f.key

                abs_f = file.split("/")[-1]

                if file.endswith(".csv"):
                    df = convert_object_to_dataframe(
                        obj=f,
                        db_name=self.db_name,
                        collection_name=self.train_data_transform_log,
                    )

                    df.fillna("NULL", inplace=True)

                    df["Wafer"] = df["Wafer"].str[6:]

                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.train_data_transform_log,
                        log_message=f"replaced  missing values with null for the file {file}",
                    )

                    df.to_csv(abs_f, index=None, header=True)

                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.train_data_transform_log,
                        log_message=f"Converted {file} to df and local copy copy is created",
                    )

                    self.s3_obj.upload_to_s3(
                        src_file=abs_f,
                        bucket=self.train_data_bucket,
                        dest_file=file,
                        db_name=self.db_name,
                        collection_name=self.train_data_transform_log,
                    )

                else:
                    pass

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_data_transform_log,
                log_message=exception_msg,
            )

            raise Exception(exception_msg)
