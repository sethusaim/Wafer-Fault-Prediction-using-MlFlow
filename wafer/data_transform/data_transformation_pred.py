
from utils.logger import App_Logger
from utils.main_utils import convert_object_to_dataframe
from utils.read_params import read_params
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class data_transform_pred:
    """
    Description :   This class shall be used for transforming the good raw training data before loading
                    it in database
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self):
        self.config = read_params()

        self.pred_data_bucket = self.config["s3_bucket"]["wafer_pred_data_bucket"]

        self.class_name = self.__class__.__name__

        self.s3_obj = S3_Operations()

        self.log_writer = App_Logger()

        self.good_pred_data_dir = self.config["data"]["pred"]["good_data_dir"]

        self.db_name = self.config["db_log"]["db_pred_log"]

        self.pred_data_transform_log = self.config["pred_db_log"]["data_transform"]

    def rename_target_column(self):
        """
        Method Name :   rename_target_column
        Description :   This method renames the target column from Good/Bad to Output.
                        We are using substring in the first column to keep only "Integer" data for ease up the
                        loading.This columns is anyways going to be removed during prediction
        Written by  :   iNeuron Intelligence
        Revisions   :   modified code based on params.yaml file
        """
        method_name = self.rename_target_column.__name__

        try:
            csv_file_objs = self.s3_obj.get_file_objects_from_s3(
                bucket=self.pred_data_bucket,
                filename=self.good_pred_data_dir,
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
                log_message="File transformation started!!",
            )

            for f in csv_file_objs:
                file = f.key

                abs_f = file.split("/")[-1]

                if file.endswith(".csv"):
                    df = convert_object_to_dataframe(
                        obj=f,
                        db_name=self.db_name,
                        collection_name=self.pred_data_transform_log,
                    )

                    df.rename(columns={"Good/Bad": "Output"}, inplace=True)

                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.pred_data_transform_log,
                        log_message=f"Renamed the output columns for the file {file}",
                    )

                    self.s3_obj.upload_df_as_csv_to_s3(
                        data_frame=df,
                        file_name=abs_f,
                        bucket=self.pred_data_bucket,
                        dest_file_name=file,
                        db_name=self.db_name,
                        collection_name=self.pred_data_transform_log,
                    )

                else:
                    pass

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
            )

    def replace_missing_with_null(self):
        """
        Method Name :   replace_missing_with_null
        Description :   This method replaces the missing values in columns with "NULL" to store in the table.
                        We are using substring in the first column to keep only "Integer" data for ease up the
                        loading.This columns is anyways going to be removed during prediction
        Written by  :   iNeuron Intelligence
        Revisions   :   modified code based on params.yaml file
        """
        method_name = self.replace_missing_with_null.__name__

        try:
            csv_file_objs = self.s3_obj.get_file_objects_from_s3(
                bucket=self.pred_data_bucket,
                filename=self.good_pred_data_dir,
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
                log_message="File transformation started",
            )

            for f in csv_file_objs:
                file = f.key

                abs_f = file.split("/")[-1]

                if file.endswith(".csv"):
                    df = convert_object_to_dataframe(
                        obj=f,
                        db_name=self.db_name,
                        collection_name=self.pred_data_transform_log,
                    )

                    df.fillna("NULL", inplace=True)

                    df["Wafer"] = df["Wafer"].str[6:]

                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.pred_data_transform_log,
                        log_message=f"Replaced missing values with null for the file {file}",
                    )

                    self.s3_obj.upload_df_as_csv_to_s3(
                        data_frame=df,
                        file_name=abs_f,
                        bucket=self.pred_data_bucket,
                        dest_file_name=file,
                        db_name=self.db_name,
                        collection_name=self.pred_data_transform_log,
                    )

                else:
                    pass

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
            )
