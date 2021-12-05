import pandas as pd
from utils.logger import App_Logger
from utils.read_params import read_params


class Data_Getter:
    """
    Description :   This class shall be used for obtaining the data from the source for training
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, db_name, logger_object):
        self.config = read_params()

        self.training_file = self.config["db_file"]["train_db_file"]

        self.db_name = db_name

        self.logger_object = logger_object

        self.log_writter = App_Logger()

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

        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.logger_object,
            log_message="Entered the get_data method of the Data_Getter class",
        )

        try:
            self.data = pd.read_csv(self.training_file)

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Data Load Successful.Exited the get_data method of the Data_Getter class",
            )

            return self.data

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message=f"Exception occured in Class : Data_Getter, Method : get_data, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Data Load Unsuccessful.Exited the get_data method of the Data_Getter class",
            )

            raise Exception(
                "Exception occured in Class : Data_Getter, Method : get_data, Error : ",
                str(e),
            )
