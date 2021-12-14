import os

import pandas as pd
from utils.logger import App_Logger
from utils.main_utils import read_params


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

        self.goodDataPath = self.config["data"]["good"]["train"]

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
            onlyfiles = [f for f in os.listdir(self.goodDataPath)]

            for file in onlyfiles:
                f = os.path.join(self.goodDataPath, file)

                csv = pd.read_csv(f)

                csv.fillna("NULL", inplace=True)

                csv["Wafer"] = csv["Wafer"].str[6:]

                file = os.path.join(self.goodDataPath, file)

                csv.to_csv(file, index=None, header=True)

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_data_transform_log,
                    log_message=" %s: File Transformed successfully!!" % file,
                )

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
