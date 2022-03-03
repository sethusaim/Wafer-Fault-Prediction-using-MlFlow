import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from utils.logger import App_Logger
from utils.read_params import read_params
from wafer.s3_bucket_operations.s3_operations import S3_Operation


class preprocessor:
    """
    Description :   This class shall be used to clean and transform the data before training
    Written by  :   iNeuron Intelligence
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self, table_name):
        self.table_name = table_name

        self.config = read_params()

        self.s3 = S3_Operation()

        self.input_files_bucket = self.config["s3_bucket"]["input_files"]

        self.null_values_file = self.config["null_values_csv_file"]

        self.knn_n_neighbors = self.config["knn_imputer"]["n_neighbors"]

        self.knn_weights = self.config["knn_imputer"]["weights"]

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

    def remove_columns(self, data, columns):
        """
        Method Name :   remove_columns
        Description :   This method removes the given columns from a pandas dataframe
        Output      :   A pandas dataframe after the removing the specified columns
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.2
        Revisions   :   Modified code based on the params.yaml file
        """
        method_name = self.remove_columns.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=self.table_name,
        )

        self.data = data

        self.columns = columns

        try:
            self.useful_data = self.data.drop(labels=self.columns, axis=1)

            self.log_writer.log(
                table_name=self.table_name, log_message="Column removal Successful",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )

            return self.useful_data

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )

    def separate_label_feature(self, data, label_column_name):
        """
        Method name :   separate_label_feature
        Description :   This method separates the features and a label columns
        Output      :   Returns two separate dataframe, one containing features and other containing labels
        On failure  :   Raise Exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.separate_label_feature.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=self.table_name,
        )

        try:
            self.X = data.drop(labels=label_column_name, axis=1)

            self.Y = data[label_column_name]

            self.log_writer.log(
                table_name=self.table_name, log_message=f"Label Separation Successful",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )

            return self.X, self.Y

        except Exception as e:
            self.log_writer.log(
                table_name=self.table_name,
                log_message="Label Separation Unsuccessful",
            )

            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )

    def is_null_present(self, data):
        """
        Method name :   is_null_present
        Description :   This method checks whether there are null values present in the pandas
                        dataframe or not
        Output      :   Returns a boolean value. True if null is present in the dataframe, False they are
                        not present
        On failure  :   1.1
        Revisions   :   moved setup to cloud
        """
        method_name = self.is_null_present.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=self.table_name,
        )

        self.null_present = False

        try:
            self.null_counts = data.isna().sum()

            for i in self.null_counts:
                if i > 0:
                    self.null_present = True
                    break

            if self.null_present:
                null_df = pd.DataFrame()

                null_df["columns"] = data.columns

                null_df["missing values count"] = np.asarray(
                    data.isna().sum()
                )

                self.s3.upload_df_as_csv(
                    data_frame=null_df,
                    local_file_name=self.null_values_file,
                    bucket_file_name=self.null_values_file,
                    bucket_name=self.input_files_bucket,
                    table_name=self.table_name
                )

            self.log_writer.log(
                table_name=self.table_name,
                log_message="Finding missing values is a success.Data written to the null values file",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )

            return self.null_present

        except Exception as e:
            self.log_writer.log(
                table_name=self.table_name, log_message="Finding missing values failed",
            )

            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )

    def impute_missing_values(self, data):
        """
        Method Name :   impute_missing_values
        Desrciption :   This method  replaces all the missing values in th dataframe using KNN imputer
        Output      :   A dataframe which has all missing values imputed
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.impute_missing_values.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=self.table_name,
        )

        self.data = data

        try:
            imputer = KNNImputer(
                n_neighbors=self.knn_n_neighbors,
                weights=self.knn_weights,
                missing_values=np.nan,
            )

            self.new_array = imputer.fit_transform(self.data)

            self.new_data = pd.DataFrame(data=self.new_array, columns=self.data.columns)

            self.log_writer.log(
                table_name=self.table_name,
                log_message=f"Imputing missing values Successful",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )

            return self.new_data

        except Exception as e:
            self.log_writer.log(
                table_name=self.table_name,
                log_message=f"Imputing missing values failed",
            )

            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )

    def get_columns_with_zero_std_deviation(self, data):
        """
        Method Name :   get_columns_with_zero_std_deviation
        Description :   This method replaces all the missing values in the dataframe using KNN imputer
        Output      :   a dataframe which has all missing values imputed
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_columns_with_zero_std_deviation.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=self.table_name,
        )

        try:
            self.data_n = data.describe()
            
            self.col_to_drop = [x for x in data.columns if self.data_n[x]["std"] == 0]

            self.log_writer.log(
                table_name=self.table_name,
                log_message="Column search for Standard Deviation of Zero Successful.",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )

            return self.col_to_drop

        except Exception as e:
            self.log_writer.log(
                table_name=self.table_name,
                log_message="Column search for Standard Deviation of Zero Failed.",
            )

            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )
