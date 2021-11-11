import os
import pickle

from utils.logger import App_Logger
from utils.read_params import read_params


class File_Operation:
    """
    Description :   This class shall be used to save the models after training and load the saved model
                    for prediction
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, db_name, logger_object):
        self.db_name = db_name

        self.logger_object = logger_object

        self.config = read_params()

        self.log_writter = App_Logger()

        self.train_models_dir = self.config["models_dir"]["trained_models_dir"]

        self.prod_model_dir = self.config["models_dir"]["prod_models_dir"]

        self.file_format = self.config["model_save_format"]

    def save_model(self, model, filename):
        """
        Method Name :   save_model
        Description :   Save model file to directory
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Output      :   model file get saved
        Revisions   :   modified code based on params.yaml file
        """

        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.logger_object,
            log_message="Entered the save_model method of the File_Operation class",
        )

        try:
            self.model_file = os.path.join(
                self.train_models_dir, filename + self.file_format
            )

            with open(file=self.model_file, mode="wb") as f:
                pickle.dump(model, f)

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Model File "
                + filename
                + " saved. Exited the save_model method of the Model_Finder class",
            )

            return "success"

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message=f"Exception occured in Class : Model_Finder, Method : save_model, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Model File "
                + filename
                + " could not be saved. Exited the save_model method of the Model_Finder class",
            )

            raise Exception(
                "Exception occured in Class : Model_Finder, Method : save_model, Error : ",
                str(e),
            )

    def load_model(self, filename):
        """
        Method Name :   load_model
        Description :   Load the model file to directory
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.logger_object,
            log_message="Entered the load_model method of the File_Operation class",
        )

        try:
            self.prod_model_file = os.path.join(
                self.prod_model_dir, filename + self.file_format
            )

            with open(file=self.prod_model_file, mode="rb") as f:
                self.log_writter.log(
                    db_name=self.db_name,
                    collection_name=self.logger_object,
                    log_message="Model File "
                    + filename
                    + " loaded. Exited the load_model method of the Model_Finder class",
                )

                return pickle.load(f)

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message=f"Exception occured in Class : Model_Finder,Method : load_model, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Model File "
                + filename
                + " could not be saved. Exited the load_model method of the Model_Finder class",
            )

            raise Exception(
                "Exception occured in Class : Model_Finder,Method : load_model, Error : ",
                str(e),
            )

    def find_correct_model_file(self, cluster_number):
        """
        Method Name :   find_correct_model_file
        Description :   select the correct model based on cluster number
        Output      :   The model file
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """

        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.logger_object,
            log_message="Entered the find_correct_model_file method of the File_Operation class",
        )

        try:
            self.cluster_number = cluster_number

            self.list_of_model_files = []

            self.list_of_files = os.listdir(self.prod_model_dir)

            for self.file in self.list_of_files:
                try:
                    if self.file.index(str(self.cluster_number)) != -1:
                        self.model_name = self.file

                except:
                    continue

            self.model_name = self.model_name.split(".")[0]

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Exited the find_correct_model_file method of the Model_Finder class.",
            )

            return self.model_name

        except Exception as e:
            self.log_writter.log(
                db_name=self.logger_object,
                log_message=f"Exception occured in Class : Model_Finder,Method : find_correct_model_file, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Exited the find_correct_model_file method of the Model_Finder class with Failure",
            )

            raise Exception(
                "Exception occured in Class : Model_Finder,Method : find_correct_model_file, Error : ",
                str(e),
            )
