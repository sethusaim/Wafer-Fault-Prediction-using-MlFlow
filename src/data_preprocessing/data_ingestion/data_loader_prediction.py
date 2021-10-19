import pandas as pd
from utils.read_params import read_params


class Data_Getter_Pred:
    """
    Description :   This class shall be used for obtaining the data from the source for prediction
    Written By  :   iNeuron Intelligence
    Version     :   1.0
    Revision    :   None
    """

    def __init__(self, file_object, logger_object):
        self.config = read_params()

        self.prediction_file = self.config["db_file"]["pred_db_file"]

        self.file_object = file_object

        self.logger_object = logger_object

    def get_data(self):
        """
        Method Name :   get_data
        Description :   This method reads the data from the source
        Written by  :   iNeuron Intelligence
        Output      :   a pandas dataframe
        On failure  :   Raise Exception
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        self.logger_object.log(
            self.file_object, "Entered the get_data method of the Data_Getter class"
        )

        try:
            self.data = pd.read_csv(self.prediction_file)

            self.logger_object.log(
                self.file_object,
                "Data Load Successful.Exited the get_data method of the Data_Getter class",
            )

            return self.data

        except Exception as e:
            self.logger_object.log(
                self.file_object,
                "Exception occured in get_data method of the Data_Getter class. Exception message: "
                + str(e),
            )

            self.logger_object.log(
                self.file_object,
                "Data Load Unsuccessful.Exited the get_data method of the Data_Getter class",
            )

            raise e
