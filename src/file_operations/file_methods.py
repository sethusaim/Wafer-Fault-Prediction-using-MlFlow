import os
import pickle

from utils.main_utils import read_params


class File_Operation:
    """
    Description :   This class shall be used to save the models after training and load the saved model
                    for prediction
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object

        self.logger_object = logger_object

        self.config = read_params()

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
        self.logger_object.log(
            self.file_object,
            "Entered the save_model method of the File_Operation class",
        )

        try:
            with open(
                self.train_models_dir + "/" + filename + self.file_format, "wb"
            ) as f:
                pickle.dump(model, f)

            self.logger_object.log(
                self.file_object,
                "Model File "
                + filename
                + " saved. Exited the save_model method of the Model_Finder class",
            )

            return "success"

        except Exception as e:
            self.logger_object.log(
                self.file_object,
                "Exception occured in save_model method of the Model_Finder class. Exception message:  "
                + str(e),
            )

            self.logger_object.log(
                self.file_object,
                "Model File "
                + filename
                + " could not be saved. Exited the save_model method of the Model_Finder class",
            )

            raise e

    def load_model(self, filename):
        """
        Method Name :   load_model
        Description :   Load the model file to directory
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        self.logger_object.log(
            self.file_object,
            "Entered the load_model method of the File_Operation class",
        )

        try:
            with open(
                self.prod_model_dir + "/" + filename + self.file_format, "rb"
            ) as f:
                self.logger_object.log(
                    self.file_object,
                    "Model File "
                    + filename
                    + " loaded. Exited the load_model method of the Model_Finder class",
                )

                return pickle.load(f)

        except Exception as e:
            self.logger_object.log(
                self.file_object,
                "Exception occured in load_model method of the Model_Finder class. Exception message:  "
                + str(e),
            )

            self.logger_object.log(
                self.file_object,
                "Model File "
                + filename
                + " could not be saved. Exited the load_model method of the Model_Finder class",
            )

            raise e

    def find_correct_model_file(self, cluster_number):
        """
        Method Name :   find_correct_model_file
        Description :   select the correct model based on cluster number
        Output      :   The model file
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        self.logger_object.log(
            self.file_object,
            "Entered the find_correct_model_file method of the File_Operation class",
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

            self.logger_object.log(
                self.file_object,
                "Exited the find_correct_model_file method of the Model_Finder class.",
            )

            return self.model_name

        except Exception as e:
            self.logger_object.log(
                self.file_object,
                "Exception occured in find_correct_model_file method of the Model_Finder class. Exception message:  "
                + str(e),
            )

            self.logger_object.log(
                self.file_object,
                "Exited the find_correct_model_file method of the Model_Finder class with Failure",
            )

            raise e
