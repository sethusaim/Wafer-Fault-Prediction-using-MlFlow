from utils.logger import App_Logger
from utils.read_params import read_params
from wafer.s3_bucket_operations.s3_operations import S3_Operation


class Data_Transform_Train:
    """
    Description :   This class shall be used for transforming the good raw training data before loading it in database
    Written by  :   iNeuron Intelligence

    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.config = read_params()

        self.train_data_bucket = self.config["s3_bucket"]["wafer_train_data"]

        self.class_name = self.__class__.__name__

        self.s3 = S3_Operation()

        self.log_writer = App_Logger()

        self.good_train_data_dir = self.config["data"]["train"]["good"]

        self.train_data_transform_log = self.config["train_db_log"]["data_transform"]

    def rename_target_column(self):
        """
        Method Name :   rename_target_column
        Description :   This method renames the target column from Good/Bad to Output.
                        We are using substring in the first column to keep only "Integer" data for ease up the
                        loading.This columns is anyways going to be removed during training

        Written by  :   iNeuron Intelligence
        Revisions   :   moved setup to cloud
        """
        method_name = self.rename_target_column.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            log_file=self.train_data_transform_log,
        )

        try:
            lst = self.s3.read_csv_from_folder(
                folder_name=self.good_train_data_dir,
                bucket=self.train_data_bucket,
                log_file=self.train_data_transform_log,
            )

            for idx, f in enumerate(lst):
                df = f[idx][0]

                file = f[idx][1]

                abs_f = f[idx][2]

                if file.endswith(".csv"):
                    df.rename(columns={"Good/Bad": "Output"}, inplace=True)

                    self.log_writer.log(
                        log_file=self.train_data_transform_log,
                        log_info=f"Renamed the output columns for the file {file}",
                    )

                    self.s3.upload_df_as_csv(
                        data_frame=df,
                        local_file_name=abs_f,
                        bucket_file_name=file,
                        bucket=self.train_data_bucket,
                        log_file=self.train_data_transform_log,
                    )

                else:
                    pass

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.train_data_transform_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.train_data_transform_log,
            )

    def replace_missing_with_null(self):
        """
        Method Name :   replace_missing_with_null
        Description :   This method replaces the missing values in columns with "NULL" to store in the table.
                        We are using substring in the first column to keep only "Integer" data for ease up the
                        loading.This columns is anyways going to be removed during training
        Written by  :   iNeuron Intelligence
        Revisions   :   moved setup to cloud
        """
        method_name = self.replace_missing_with_null.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            log_file=self.train_data_transform_log,
        )

        try:
            lst = self.s3.read_csv_from_folder(
                folder_name=self.good_train_data_dir,
                bucket=self.train_data_bucket,
                log_file=self.train_data_transform_log,
            )

            for idx, f in enumerate(lst):
                df = f[idx][0]

                file = f[idx][1]

                abs_f = f[idx][2]

                if file.endswith(".csv"):
                    df.fillna("NULL", inplace=True)

                    df["Wafer"] = df["Wafer"].str[6:]

                    self.log_writer.log(
                        log_file=self.train_data_transform_log,
                        log_info=f"Replaced missing values with null for the file {file}",
                    )

                    self.s3.upload_df_as_csv(
                        data_frame=df,
                        local_file_name=abs_f,
                        bucket_file_name=file,
                        bucket=self.train_data_bucket,
                        log_file=self.train_data_transform_log,
                    )

                else:
                    pass

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.train_data_transform_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.train_data_transform_log,
            )
