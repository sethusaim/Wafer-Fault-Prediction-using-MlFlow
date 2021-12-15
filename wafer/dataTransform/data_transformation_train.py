import os

import pandas as pd
from utils.logger import App_Logger
from utils.main_utils import make_readable, read_params
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

    def replaceMissingWithNull(self):
        """
        Method Name :   replaceMissingWithNull
        Description :   This method replaces the missing values in columns with "NULL" to store in the table.
                        We are using substring in the first column to keep only "Integer" data for ease up the
                        loading.This columns is anyways going to be removed during training
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        try:
            csv_file_objs = self.s3_obj.get_csv_objs_from_s3(
                bucket=self.good_data_bucket
            )

            for f in csv_file_objs:
                file = f.key

                file_content = f.get()["Body"].read().decode()

                data = make_readable(file_content)

                csv = pd.read_csv(data)

                csv.fillna("NULL", inplace=True)

                csv["Wafer"] = csv["Wafer"].str[6:]

                csv.to_csv(file, index=None, header=True)

                self.s3_obj.upload_to_s3(
                    src_file=file, bucket=self.good_data_bucket, dest_file=file
                )

                os.remove(file)

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_data_transform_log,
                log_message=f"Exception occured in Class : dataTransform, Method : replaceMissingWithNull, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : dataTransform, Method : replaceMissingWithNull, Error : ",
                str(e),
            )
