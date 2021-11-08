import os
from os import listdir

import pandas
from utils.application_logging.logger import App_Logger
from utils.main_utils import read_params


class dataTransformPredict:
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

        self.pred_data_transform_log = os.path.join(
            self.config["log_dir"]["pred_log_dir"], "dataTransformLog.txt"
        )

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
            log_file = open(self.pred_data_transform_log, "a+")

            onlyfiles = [f for f in listdir(self.goodDataPath)]

            for file in onlyfiles:
                csv = pandas.read_csv(self.goodDataPath + "/" + file)

                csv.fillna("NULL", inplace=True)

                csv["Wafer"] = csv["Wafer"].str[6:]

                csv.to_csv(self.goodDataPath + "/" + file, index=None, header=True)

                self.logger.log(log_file, " %s: File Transformed successfully!!" % file)

        except Exception as e:
            self.logger.log(log_file, "Data Transformation failed because:: %s" % e)

            log_file.close()

            raise e

        log_file.close()
