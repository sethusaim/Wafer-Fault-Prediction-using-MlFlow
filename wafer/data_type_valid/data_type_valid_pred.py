from utils.exception import raise_exception
from utils.logger import App_Logger
from utils.main_utils import convert_object_to_dataframe
from utils.read_params import read_params
from wafer.mongo_db_operations.db_operations import MongoDBOperation
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class dBOperation:
    """
    Description :    This class shall be used for handling all the db operations
    Written by  :    iNeuron Intelligence
    Version     :    1.0
    Revisions   :    None
    """

    def __init__(self):
        self.config = read_params()

        self.s3_obj = S3_Operations()

        self.db_op = MongoDBOperation()

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

        self.pred_data_bucket = self.config["s3_bucket"]["wafer_pred_data_bucket"]

        self.good_data_pred_dir = self.config["data"]["pred"]["good_data_dir"]

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.db_name = self.config["db_log"]["db_pred_log"]

        self.pred_db_insert_log = self.config["pred_db_log"]["db_insert"]

        self.pred_export_csv_log = self.config["pred_db_log"]["export_csv"]

    def insert_good_data_as_record(self, db_name, collection_name):
        method_name = self.insert_good_data_as_record.__name__

        try:
            csv_files = self.s3_obj.get_file_objects_from_s3(
                bucket=self.pred_data_bucket,
                filename=self.good_data_pred_dir,
                db_name=self.db_name,
                collection_name=self.pred_db_insert_log,
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_db_insert_log,
                log_message="Got csv objects from s3 bucket",
            )

            for f in csv_files:
                file = f.key

                if file.endswith(".csv"):
                    df = convert_object_to_dataframe(
                        obj=f,
                        db_name=self.db_name,
                        collection_name=self.pred_db_insert_log,
                    )

                    self.db_op.insert_dataframe_as_record(
                        data_frame=df,
                        db_name=db_name,
                        collection_name=collection_name,
                    )

                else:
                    pass

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.pred_db_insert_log,
                    log_message="Inserted dataframe as collection record in mongodb",
                )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_db_insert_log,
            )

    def export_collection_to_csv(self, db_name, collection_name):
        method_name = self.export_collection_to_csv.__name__

        try:
            df = self.db_op.convert_collection_to_dataframe(
                db_name=db_name, collection_name=collection_name
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_export_csv_log,
                log_message="Got the collection as dataframe",
            )

            csv_file = self.config["export_pred_csv_file"]

            df.to_csv(csv_file, index=False, header=True)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_export_csv_log,
                log_message="Dataframe is converted to csv file and local copy is created",
            )

            self.s3_obj.upload_to_s3(
                src_file=csv_file,
                bucket=self.input_files_bucket,
                dest_file=csv_file,
                db_name=self.db_name,
                collection_name=self.pred_export_csv_log,
            )

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )
