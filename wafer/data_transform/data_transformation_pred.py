import os

import pandas
from utils.logger import App_Logger
from utils.read_params import read_params


class data_transformPredict:
    """
    Description :   This class shall be used for transforming the good raw training data before loading
                    it in database
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self):
        self.config = read_params()

        self.goodDataPath = self.config["data"]["good"]["pred"]

        self.logger = App_Logger()

        self.db_name = self.config["db_log"]["db_pred_log"]

        self.pred_data_transform_log = self.config["pred_db_log"]["data_transform"]

    def replaceMissingWithNull(self):
        """
        Method Name :   replaceMissingWithNull
        Description :   This method replaces the missing values in columns with "NULL" to store in the table.
                        We are using substring in the first column to keep only "Integer" data for ease up the
                        loading.This columns is anyways going to be removed during prediction
        Written by  :   iNeuron Intelligence
        Revisions   :   modified code based on params.yaml file
        """
        try:
            onlyfiles = [f for f in os.listdir(self.goodDataPath)]

            for file in onlyfiles:
                csv = pandas.read_csv(self.goodDataPath + "/" + file)

                csv.fillna("NULL", inplace=True)

                csv["Wafer"] = csv["Wafer"].str[6:]

                csv.to_csv(self.goodDataPath + "/" + file, index=None, header=True)

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.pred_data_transform_log,
                    log_message=" %s: File Transformed successfully!!" % file,
                )

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
                log_message=f"Exception Occured in Class : data_transformPredict.\
                    Method : replaceMissingWithNull, Error : {str(e)} ",
            )

            raise Exception(
                "Exception Occured in Class : data_transformPredict.\
                    Method : replaceMissingWithNull, Error : ",
                str(e),
            )
