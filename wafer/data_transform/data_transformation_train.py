from utils.logger import app_logger
from utils.read_params import read_params
from wafer.s3_bucket_operations.s3_operations import s3_operations


class data_transform_train:
    """
    Description :   This class shall be used for transforming the good raw trainiction data before loading
                    it in database
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self):
        self.config = read_params()

        self.train_data_bucket = self.config["s3_bucket"]["wafer_train_data_bucket"]

        self.class_name = self.__class__.__name__

        self.s3 = s3_operations()

        self.log_writer = app_logger()

        self.good_train_data_dir = self.config["data"]["train"]["good_data_dir"]

        self.db_name = self.config["db_log"]["db_train_log"]

        self.train_data_transform_log = self.config["train_db_log"]["data_transform"]

    def rename_target_column(self):
        """
        Method Name :   rename_target_column
        Description :   This method renames the target column from Good/Bad to Output.
                        We are using substring in the first column to keep only "Integer" data for ease up the
                        loading.This columns is anyways going to be removed during trainiction
        Written by  :   iNeuron Intelligence
        Revisions   :   modified code based on params.yaml file
        """
        method_name = self.rename_target_column.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            collection_name=self.train_data_transform_log,
        )

        try:
            lst = self.s3.read_csv(
                bucket=self.train_data_bucket,
                file_name=self.good_train_data_dir,
                folder=True,
                table_name=self.train_data_transform_log,
            )

            for idx, f in enumerate(lst):
                df = f[idx][0]

                file = f[idx][1]

                abs_f = f[idx][2]

                if file.endswith(".csv"):
                    df.rename(columns={"Good/Bad": "Output"}, inplace=True)

                    self.log_writer.log(
                        collection_name=self.train_data_transform_log,
                        log_message=f"Renamed the output columns for the file {file}",
                    )

                    self.s3.upload_df_as_csv(
                        data_frame=df,
                        file_name=abs_f,
                        bucket=self.train_data_bucket,
                        dest_file_name=file,
                        collection_name=self.train_data_transform_log,
                    )

                else:
                    pass

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                collection_name=self.train_data_transform_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                collection_name=self.train_data_transform_log,
            )

    def replace_missing_with_null(self):
        """
        Method Name :   replace_missing_with_null
        Description :   This method replaces the missing values in columns with "NULL" to store in the table.
                        We are using substring in the first column to keep only "Integer" data for ease up the
                        loading.This columns is anyways going to be removed during trainiction
        Written by  :   iNeuron Intelligence
        Revisions   :   modified code based on params.yaml file
        """
        method_name = self.replace_missing_with_null.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            collection_name=self.train_data_transform_log,
        )

        try:
            lst = self.s3.read_csv(
                bucket=self.train_data_bucket,
                file_name=self.good_train_data_dir,
                folder=True,
                table_name=self.train_data_transform_log,
            )

            for idx, f in enumerate(lst):
                df = f[idx][0]

                file = f[idx][1]

                abs_f = f[idx][2]

                if file.endswith(".csv"):
                    df.fillna("NULL", inplace=True)

                    df["Wafer"] = df["Wafer"].str[6:]

                    self.log_writer.log(
                        collection_name=self.train_data_transform_log,
                        log_message=f"Replaced missing values with null for the file {file}",
                    )

                    self.s3.upload_df_as_csv(
                        data_frame=df,
                        file_name=abs_f,
                        bucket=self.train_data_bucket,
                        dest_file_name=file,
                        collection_name=self.train_data_transform_log,
                    )

                else:
                    pass

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                collection_name=self.train_data_transform_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                collection_name=self.train_data_transform_log,
            )
