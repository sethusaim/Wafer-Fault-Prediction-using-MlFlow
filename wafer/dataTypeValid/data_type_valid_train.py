import os

from utils.logger import App_Logger
from utils.main_utils import convert_object_to_dataframe
from utils.read_params import read_params
from wafer.mongo_db_operations.db_operations import MongoDBOperation
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class dBOperation:
    """
    Description :    This class shall be used for handling all the SQL operations
    Written by  :    iNeuron Intelligence
    Version     :    1.0
    Revisions   :    None
    """

    def __init__(self):
        self.config = read_params()

        self.badFilePath = self.config["data"]["bad"]["train"]

        self.goodFilePath = self.config["data"]["good"]["train"]

        self.s3_obj = S3_Operations()

        self.db_op = MongoDBOperation()

        self.logger = App_Logger()

        self.bad_data_bucket = self.config["s3_bucket"]["data_bad_train_bucket"]

        self.good_data_bucket = self.config["s3_bucket"]["data_good_train_bucket"]

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.db_name = self.config["db_log"]["db_train_log"]

        self.train_db_insert_log = self.config["train_db_log"]["db_insert"]

        self.train_export_csv_log = self.config["train_db_log"]["export_csv"]

    def insert_good_data_as_record(self, db_name, collection_name):
        try:
            csv_files = self.s3_obj.get_file_objs_from_s3(bucket=self.good_data_bucket)

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_db_insert_log,
                log_message="Got csv objects from s3 bucket",
            )

            for f in csv_files:
                file_name = f.key

                df = convert_object_to_dataframe(f)

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_db_insert_log,
                    log_message=f"Converted {file_name} to dataframe",
                )

                self.db_op.insert_dataframe_as_record(
                    db_name=db_name,
                    collection_name=collection_name,
                    data_frame=df,
                )

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_db_insert_log,
                    log_message=f"Inserted {file_name} data as record to mongodb",
                )

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_db_insert_log,
                log_message="Inserted dataframe as collection record in mongodb",
            )

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_export_csv_log,
                log_message=f"Exception occured in Class : dbOperation, Method : insert_good_data_as_record, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : dbOperation, Method : insert_good_data_as_record, Error : ",
                str(e),
            )

    def export_collection_to_csv(self, db_name, collection_name):
        try:
            df = self.db_op.convert_collection_to_dataframe(
                db_name=db_name, collection_name=collection_name
            )

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_export_csv_log,
                log_message="Got the collection as dataframe",
            )

            csv_file = self.config["export_train_csv_file"]

            df.to_csv(csv_file, index=False, header=True)

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_export_csv_log,
                log_message="Dataframe is converted to csv file and local copy is created",
            )

            self.s3_obj.upload_to_s3(
                src_file=csv_file,
                bucket=self.input_files_bucket,
                dest_file=csv_file,
            )

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_export_csv_log,
                log_message=f"csv file uploaded to {self.input_files_bucket}",
            )

            os.remove(csv_file)

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_export_csv_log,
                log_message="local copy of the csv file is removed",
            )

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_export_csv_log,
                log_message=f"Exception occured in Class : dbOperation, Method : export_collection_to_csv, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : dbOperation, Method : export_collection_to_csv, Error : ",
                str(e),
            )
