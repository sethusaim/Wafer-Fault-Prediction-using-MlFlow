import os

from utils.logger import App_Logger
from utils.main_utils import convert_object_to_dataframe, read_params
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class dataTransform:
    """
    Description :   This class shall be used for transforming the Good Raw Training data before loaded
                    it in database
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self):
        self.config = read_params()

        self.good_data_bucket = self.config["s3_bucket"]["data_good_train_bucket"]

        self.s3_obj = S3_Operations()

        self.logger = App_Logger()

        self.db_name = self.config["db_log"]["db_train_log"]

        self.train_data_transform_log = self.config["train_db_log"]["data_transform"]

    def rename_target_column(self):
        """
        Method Name :   replace_missing_with_null
        Description :   This method replaces the missing values in columns with "NULL" to store in the table.
                        We are using substring in the first column to keep only "Integer" data for ease up the
                        loading.This columns is anyways going to be removed during training
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """

        try:
            csv_file_objs = self.s3_obj.get_file_objs_from_s3(
                bucket=self.good_data_bucket
            )

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_data_transform_log,
                log_message=f"Got df objects from s3 bucket : {self.good_data_bucket}",
            )

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_data_transform_log,
                log_message="File transformation started",
            )

            for f in csv_file_objs:
                file = f.key

                df = convert_object_to_dataframe(f)

                df.rename(columns={"Good/Bad": "Output"}, inplace=True)

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_data_transform_log,
                    log_message=f"Renamed the output column for the file for the file {file} ",
                )

                df.to_csv(file, index=None, header=True)

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_data_transform_log,
                    log_message=f"Converted {file} to df and local copy copy is created",
                )

                self.s3_obj.upload_to_s3(
                    src_file=file, bucket=self.good_data_bucket, dest_file=file
                )

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_data_transform_log,
                    log_message=f"Uploaded {file} to s3 bucket : {self.good_data_bucket}",
                )

                os.remove(file)

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_data_transform_log,
                    log_message=f"Removed the local copy of {file}",
                )

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_data_transform_log,
                log_message=f"Exception occured in Class : dataTransform, Method : replace_missing_with_null, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : dataTransform, Method : replace_missing_with_null, Error : ",
                str(e),
            )

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
        try:
            csv_file_objs = self.s3_obj.get_file_objs_from_s3(
                bucket=self.good_data_bucket
            )

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_data_transform_log,
                log_message=f"Got df objects from s3 bucket : {self.good_data_bucket}",
            )

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_data_transform_log,
                log_message="File transformation started",
            )

            for f in csv_file_objs:
                file = f.key

                df = convert_object_to_dataframe(f)

                df.fillna("NULL", inplace=True)

                df["Wafer"] = df["Wafer"].str[6:]

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_data_transform_log,
                    log_message=f"replaced  missing values with null for the file {file}",
                )

                df.to_csv(file, index=None, header=True)

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_data_transform_log,
                    log_message=f"Converted {file} to df and local copy copy is created",
                )

                self.s3_obj.upload_to_s3(
                    src_file=file, bucket=self.good_data_bucket, dest_file=file
                )

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_data_transform_log,
                    log_message=f"Uploaded {file} to s3 bucket : {self.good_data_bucket}",
                )

                os.remove(file)

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_data_transform_log,
                    log_message=f"Removed the local copy of {file}",
                )

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_data_transform_log,
                log_message=f"Exception occured in Class : dataTransform, Method : replace_missing_with_null, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : dataTransform, Method : replace_missing_with_null, Error : ",
                str(e),
            )
