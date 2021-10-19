import pandas as pd
from utils.read_params import read_params


class Data_Getter:
    """
    Description :   This class shall be used for obtaining the data from the source for training
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, file_object, logger_object):
        self.config = read_params()

        self.training_file = self.config["db_file"]["train_db_file"]

        self.file_object = file_object

        self.logger_object = logger_object

    def get_data(self):
        """
        Method Name :   get_data
        Description :   This method reads the data from the source
        Output      :   A pandas dataframe
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        self.logger_object.log(
            self.file_object, "Entered the get_data method of the Data_Getter class"
        )

        try:
            self.data = pd.read_csv(self.training_file)

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
