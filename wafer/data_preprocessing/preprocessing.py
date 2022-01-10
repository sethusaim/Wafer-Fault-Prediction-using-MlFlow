import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from utils.exception import raise_exception_log
from utils.logger import App_Logger
from utils.read_params import read_params
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class Preprocessor:
    """
    Description :   This class shall be used to clean and transform the data before training
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, db_name, collection_name):
        self.collection_name = collection_name

        self.db_name = db_name

        self.config = read_params()

        self.s3_obj = S3_Operations()

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.null_values_file = self.config["null_values_csv_file"]

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

    def remove_columns(self, data, columns):
        """
        Method Name :   remove_columns
        Description :   This method removes the given columns from a pandas dataframe
        Output      :   A pandas dataframe after the removing the specified columns
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   Modified code based on the params.yaml file
        """
        method_name = self.remove_columns.__name__

        self.log_writer.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message=f"Entered the {method_name} method of the {self.class_name} class",
        )

        self.data = data

        self.columns = columns

        try:
            self.useful_data = self.data.drop(labels=self.columns, axis=1)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Column removal Successful.Exited the {method_name} method of the {self.class_name} class",
            )

            return self.useful_data

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Column removal Unsuccessful. Exited the {method_name} method of the {self.class_name} class",
            )

            raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def separate_label_feature(self, data, label_column_name):
        """
        Method name :   separate_label_feature
        Description :   This method separates the features and a label columns
        Output      :   Returns two separate dataframe, one containing features and other containing labels
        On failure  :   Raise Exception
        Version     :   1.1
        Revisions   :   modified code based on the params.yaml file
        """
        method_name = self.separate_label_feature.__name__

        self.log_writer.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message=f"Entered the {method_name} method of the {self.class_name} class",
        )

        try:
            self.X = data.drop(labels=label_column_name, axis=1)

            self.Y = data[label_column_name]

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Label Separation Successful. Exited the {method_name} method of the {self.class_name} class",
            )

            return self.X, self.Y

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Label Separation Unsuccessful.Exited the {method_name} method of the {self.class_name} class",
            )

            raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def is_null_present(self, data):
        """
        Method name :   is_null_present
        Description :   This method checks whether there are null values present in the pandas
                        dataframe or not
        Output      :   Returns a boolean value. True if null is present in the dataframe, False they are
                        not present
        On failure  :   1.1
        Revisions   :   modified code based on the params.yaml file
        """
        method_name = self.is_null_present.__name__

        self.log_writer.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message=f"Entered the {method_name} method of the {self.class_name} class",
        )

        self.null_present = False

        try:
            self.null_counts = data.isna().sum()

            for i in self.null_counts:
                if i > 0:
                    self.null_present = True
                    break

            if self.null_present:
                dataframe_with_null = pd.DataFrame()

                dataframe_with_null["columns"] = data.columns

                dataframe_with_null["missing values count"] = np.asarray(
                    data.isna().sum()
                )

                self.s3_obj.upload_df_as_csv_to_s3(
                    data_frame=dataframe_with_null,
                    file_name=self.null_values_file,
                    bucket=self.input_files_bucket,
                    dest_file_name=self.null_values_file,
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Finding missing values is a success.Data written to the null values file.  \
                    Exited the {method_name} method of the {self.class_name} class",
            )

            return self.null_present

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Finding missing values failed. Exited the {method_name} method of the {self.class_name} class",
            )

            raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def impute_missing_values(self, data):
        """
        Method Name :   impute_missing_values
        Desrciption :   This method  replaces all the missing values in th dataframe using KNN imputer
        Output      :   A dataframe which has all missing values imputed
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on the params.yaml file
        """

        method_name = self.impute_missing_values.__name__

        self.log_writer.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message=f"Entered the {method_name} method of the {self.class_name} class",
        )

        self.data = data

        try:
            imputer = KNNImputer(
                n_neighbors=self.config["knn_imputer"]["n_neighbors"],
                weights=self.config["knn_imputer"]["weights"],
                missing_values=np.nan,
            )

            self.new_array = imputer.fit_transform(self.data)

            self.new_data = pd.DataFrame(data=self.new_array, columns=self.data.columns)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Imputing missing values Successful. \
                    Exited the {method_name} method of the {self.class_name} class",
            )

            return self.new_data

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Imputing missing values failed. Exited the {method_name} method of the {self.class_name} class",
            )

            raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def get_columns_with_zero_std_deviation(self, data):
        """
        Method Name :   get_columns_with_zero_std_deviation
        Description :   This method replaces all the missing values in the dataframe using KNN imputer
        Output      :   a dataframe which has all missing values imputed
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """

        method_name = self.get_columns_with_zero_std_deviation.__name__

        self.log_writer.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message=f"Entered the {method_name} method of the {self.class_name} class",
        )

        self.data_n = data.describe()

        try:
            self.col_to_drop = [x for x in data.columns if self.data_n[x]["std"] == 0]

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Column search for Standard Deviation of Zero Successful.Exited the {method_name} method of the {self.class_name} class",
            )

            return self.col_to_drop

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Column search for Standard Deviation of Zero Failed. Exited the {method_name} method of the {self.class_name} class",
            )

            raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )
