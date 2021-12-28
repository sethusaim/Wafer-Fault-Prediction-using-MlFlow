import os

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from utils.logger import App_Logger
from utils.main_utils import read_params
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

        self.bad_data_bucket = self.config["s3_bucket"]["data_bad_train_bucket"]

        self.log_writter = App_Logger()

    def remove_columns(self, data, columns):
        """
        Method Name :   remove columns
        Description :   This method removes the given columns from a pandas dataframe
        Output      :   A pandas dataframe after the removing the specified columns
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   Modified code based on the params.yaml file
        """

        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the remove_columns method of the Preprocessor class",
        )

        self.data = data

        self.columns = columns

        try:
            self.useful_data = self.data.drop(labels=self.columns, axis=1)

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Column removal Successful.Exited the remove_columns method of the Preprocessor class",
            )

            return self.useful_data

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Preprocessor, Method : remove_columns, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class",
            )

            raise Exception(
                "Exception occured in Class : Preprocessor, Method : remove_columns, Error : ",
                str(e),
            )

    def separate_label_feature(self, data, label_column_name):
        """
        Method name :   separate_label_features
        Description :   This method separates the features and a label columns
        Output      :   Returns two separate dataframe, one containing features and other containing labels
        On failure  :   Raise Exception
        Version     :   1.1
        Revisions   :   modified code based on the params.yaml file
        """

        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the separate_label_feature method of the Preprocessor class",
        )

        try:
            self.X = data.drop(labels=label_column_name, axis=1)

            self.Y = data[label_column_name]

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Label Separation Successful. \
                    Exited the separate_label_feature method of the Preprocessor class",
            )

            return self.X, self.Y

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Preprocessor.\
                    Method : separate_label_feature method, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Label Separation Unsuccessful. \
                    Exited the separate_label_feature method of the Preprocessor class",
            )

            raise Exception(
                "Exception occured in Class : Preprocessor, Method : separate_label_feature method, Error :",
                str(e),
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
        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the is_null_present method of the Preprocessor class",
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

                null_values_file = self.config["null_values_csv_file"]

                dataframe_with_null.to_csv(null_values_file)

                self.log_writter.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_message="Prepared the null values csv file and created a local copy of the same",
                )

                self.s3_obj.upload_to_s3(
                    src_file=null_values_file,
                    bucket=self.bad_data_bucket,
                    dest_file=null_values_file,
                )

                self.log_writter.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_message=f"Upload the {null_values_file} to {self.bad_data_bucket} bucket",
                )

                os.remove(null_values_file)

                self.log_writter.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_message=f"Local copy of {null_values_file} is deleted",
                )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Finding missing values is a success.Data written to the null values file.  \
                    Exited the is_null_present method of the Preprocessor class",
            )

            return self.null_present

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Preprocessor, Method : is_null_present, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Finding missing values failed. Exited the is_null_present method of the Preprocessor class",
            )

            raise Exception(
                "Exception occured in Class : Preprocessor, Method : is_null_present, Error : ",
                str(e),
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

        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the impute_missing_values method of the Preprocessor class",
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

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Imputing missing values Successful. \
                    Exited the impute_missing_values method of the Preprocessor class",
            )

            return self.new_data

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Preprocessor. \
                    Method : impute_missing_values, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class",
            )

            raise Exception(
                "Exception occured in Class : Preprocessor, Method : impute_missing_values, Error : ",
                str(e),
            )

    def get_columns_with_zero_std_deviation(self, data):
        """
        Method Name :   get_colums_with_zero_std_deviation
        Description :   This method replaces all the missing values in the dataframe using KNN imputer
        Output      :   a dataframe which has all missing values imputed
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the get_columns_with_zero_std_deviation method of the Preprocessor class",
        )

        self.columns = data.columns

        self.data_n = data.describe()

        self.col_to_drop = []

        try:
            for x in self.columns:
                if self.data_n[x]["std"] == 0:
                    self.col_to_drop.append(x)

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Column search for Standard Deviation of Zero Successful \
                Exited the get_columns_with_zero_std_deviation method of the Preprocessor class",
            )

            return self.col_to_drop

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Preprocessor, Method : get_columns_with_zero_std_deviation. \
                     Error : {str(e)} ",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Column search for Standard Deviation of Zero Failed. \
                    Exited the get_columns_with_zero_std_deviation method of the Preprocessor class",
            )

            raise Exception(
                "Exception occured in Class : Preprocessor, Method : get_columns_with_zero_std_deviation. \
                     Error : ",
                str(e),
            )
