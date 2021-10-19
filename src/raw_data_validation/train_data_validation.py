import json
import os
import re
import shutil
from datetime import datetime
from os import listdir

import pandas as pd
from utils.application_logging.logger import App_Logger
from utils.read_params import read_params


class Raw_Data_validation:
    """
    Description :   This class shall be used for validating the training raw data
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, path):
        self.Batch_Directory = path

        self.config = read_params()

        self.schema_path = self.config["schema_dir"]["train_schema_file"]

        self.logger = App_Logger()

        self.train_values_from_schema_log = os.path.join(
            self.config["log_dir"]["train_log_dir"], "valuesfromSchemaValidationLog.txt"
        )

        self.train_gen_log = os.path.join(
            self.config["log_dir"]["train_log_dir"], "GeneralLog.txt"
        )

        self.train_name_valid_log = os.path.join(
            self.config["log_dir"]["train_log_dir"], "nameValidationLog.txt"
        )

        self.train_col_valid_log = os.path.join(
            self.config["log_dir"]["train_log_dir"], "columnValidationLog.txt"
        )

        self.train_missing_value_log = os.path.join(
            self.config["log_dir"]["train_log_dir"], "missingValuesInColumn.txt"
        )

    def valuesFromSchema(self):
        """
        Method Name :   valuesFromSchema
        Description :   This method extracts all the relevant information from the pre defined schema file
        Output      :   LengthOfDateStampInFile,LengthOfTimeStampInFile,column_names,NumberofColumns,
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        try:
            with open(self.schema_path, "r") as f:
                dic = json.load(f)

                f.close()

            LengthOfDateStampInFile = dic["LengthOfDateStampInFile"]

            LengthOfTimeStampInFile = dic["LengthOfTimeStampInFile"]

            column_names = dic["ColName"]

            NumberofColumns = dic["NumberofColumns"]
            file = open(self.train_values_from_schema_log, "a+")

            message = (
                "LengthOfDateStampInFile:: %s" % LengthOfDateStampInFile
                + "\t"
                + "LengthOfTimeStampInFile:: %s" % LengthOfTimeStampInFile
                + "\t "
                + "NumberofColumns:: %s" % NumberofColumns
                + "\n"
            )

            self.logger.log(file, message)

            file.close()

        except ValueError:
            file = open(self.train_values_from_schema_log, "a+")

            self.logger.log(
                file, "ValueError:Value not found inside schema_training.json"
            )

            file.close()

            raise ValueError

        except KeyError:
            file = open(self.train_values_from_schema_log, "a+")

            self.logger.log(file, "KeyError:Key value error incorrect key passed")

            file.close()

            raise KeyError

        except Exception as e:
            file = open(self.train_values_from_schema_log, "a+")

            self.logger.log(file, str(e))

            file.close()

            raise e

        return (
            LengthOfDateStampInFile,
            LengthOfTimeStampInFile,
            column_names,
            NumberofColumns,
        )

    def manualRegexCreation(self):
        """
        Method Name :   manualRegexCreation
        Description :   This method contains a manually defined regex based on the filename given in
                        the schema file
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        try:
            regex = "['wafer']+['\_'']+[\d_]+[\d]+\.csv"

            return regex

        except Exception as e:
            file = open(self.train_gen_log, mode="a+")

            self.logger.log(file, "Error Occured in regex creation " + str(e))

            file.close()

            raise e

    def createDirectoryForGoodBadRawData(self):
        """
        Method Name :   createDirectoryForGoodBadRawData
        Description :   This method creates a directories to store the good data and bad data after
                        validating the prediction data
        Written by  :   iNeuron Intelligence
        On Failure  :   OS error
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        try:
            path = self.config["data"]["good"]["train"]

            if not os.path.isdir(path):
                os.makedirs(path)

            path = self.config["data"]["bad"]["train"]

            if not os.path.isdir(path):
                os.makedirs(path)

        except Exception as ex:
            file = open(self.train_gen_log, "a+")

            self.logger.log(file, "Error while creating Directory %s:" % ex)

            file.close()

            raise ex

    def deleteExistingGoodDataTrainingFolder(self):
        """
        Method Name :   deleteExistingGoodDataTrainingFolder
        Description :   This method deletes the directory made to store the good data after loading
                        the data in the table. Once the good files are loaded in the DB,
                        deleting the directory ensures space optimization
        Written by  :   iNeuron Intelligence
        On Failure  :   OS error
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        try:
            path = self.config["data"]["good"]["train"]

            if os.path.isdir(path):
                shutil.rmtree(path)

                file = open(self.train_gen_log, "a+")

                self.logger.log(file, "GoodRaw directory deleted successfully!!!")

                file.close()

        except Exception as s:
            file = open(self.train_gen_log, "a+")

            self.logger.log(file, "Error while Deleting Directory : %s" % s)

            file.close()

            raise s

    def deleteExistingBadDataTrainingFolder(self):
        """
        Method Name :   deleteExistingBadDataTrainingFolder
        Description :   This method deletes the directory made to store the good data after loading
                        the data in the table. Once the good files are loaded in the DB,deleting the directory
                        ensure space optimization
        Written by  :   iNeuron Intelligence
        On Failure  :   OS error
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        try:
            path = self.config["data"]["bad"]["train"]

            if os.path.isdir(path):
                shutil.rmtree(path)

                file = open(self.train_gen_log, "a+")

                self.logger.log(
                    file, "BadRaw directory deleted before starting validation!!!"
                )

                file.close()

        except Exception as s:
            file = open(self.train_gen_log, "a+")

            self.logger.log(file, "Error while Deleting Directory : %s" % s)

            file.close()

            raise s

    def moveBadFilesToArchiveBad(self):
        """
        Method Name :   moveBadFilesToArchiveBad
        Description :   This method deletes the directory made to store the bad data after moving the data
                        in an archive folder. We archive the bad files to send them back to the client for
                        invalid data issue
        On Failure  :   OS error
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        now = datetime.now()

        date = now.date()

        time = now.strftime("%H%M%S")

        try:
            source = self.config["data"]["bad"]["train"]

            if os.path.isdir(source):
                path = self.config["data"]["archived"]["train"]

                if not os.path.isdir(path):
                    os.makedirs(path)

                dest = os.path.join(
                    self.config["data"]["archived"]["train"],
                    "BadData_" + str(date) + "_" + str(time),
                )

                if not os.path.isdir(dest):
                    os.makedirs(dest)

                files = os.listdir(source)

                for f in files:
                    if f not in os.listdir(dest):
                        shutil.move(source + "/" + f, dest)

                file = open(self.train_gen_log, "a+")

                self.logger.log(file, "Bad files moved to archive")

                path = self.config["data"]["bad"]["train"]

                if os.path.isdir(path):
                    shutil.rmtree(path)

                self.logger.log(file, "Bad Raw Data Folder Deleted successfully!!")

                file.close()

        except Exception as e:
            file = open(self.train_gen_log, "a+")

            self.logger.log(file, "Error while moving bad files to archive:: %s" % e)

            file.close()

            raise e

    def validationFileNameRaw(
        self, regex, LengthOfDateStampInFile, LengthOfTimeStampInFile
    ):
        """
        Method Name :   validationFileNameRaw
        Description :   This function validates the name of the prediction csv file as per the given name
                        in the schema. Regex pattern is used to do the validation if name format do not match
                        the file is moved to bad raw data folder else in good raw data folder
        On Failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        self.deleteExistingBadDataTrainingFolder()

        self.deleteExistingGoodDataTrainingFolder()

        self.createDirectoryForGoodBadRawData()

        onlyfiles = [f for f in listdir(self.Batch_Directory)]

        try:
            f = open(self.train_name_valid_log, "a+")

            for filename in onlyfiles:
                if re.match(regex, filename):
                    splitAtDot = re.split(".csv", filename)
                    splitAtDot = re.split("_", splitAtDot[0])

                    if len(splitAtDot[1]) == LengthOfDateStampInFile:

                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:
                            shutil.copy(
                                self.config["data_source"]["train_data_source"]
                                + "/"
                                + filename,
                                self.config["data"]["good"]["train"],
                            )

                            self.logger.log(
                                f,
                                "Valid File name!! File moved to GoodRaw Folder :: %s"
                                % filename,
                            )

                        else:
                            shutil.copy(
                                self.config["data_source"]["train_data_source"]
                                + "/"
                                + filename,
                                self.config["data"]["bad"]["train"],
                            )

                            self.logger.log(
                                f,
                                "Invalid File Name!! File moved to Bad Raw Folder :: %s"
                                % filename,
                            )

                    else:
                        shutil.copy(
                            self.config["data_source"]["train_data_source"]
                            + "/"
                            + filename,
                            self.config["data"]["bad"]["train"],
                        )

                        self.logger.log(
                            f,
                            "Invalid File Name!! File moved to Bad Raw Folder :: %s"
                            % filename,
                        )

                else:
                    shutil.copy(
                        self.config["data_source"]["train_data_source"]
                        + "/"
                        + filename,
                        self.config["data"]["bad"]["train"],
                    )

                    self.logger.log(
                        f,
                        "Invalid File Name!! File moved to Bad Raw Folder :: %s"
                        % filename,
                    )

            f.close()

        except Exception as e:
            f = open(self.train_name_valid_log, "a+")

            self.logger.log(f, "Error occured while validating FileName %s" % e)

            f.close()

            raise e

    def validateColumnLength(self, NumberofColumns):
        """
        Method Name :   validateColumnLength
        Description :   This function validates the number of columns in the csv files. It should be same
                        as given in the schema file. If not same file is not suitable for processing and
                        thus is moved to baw raw data folder. If the column number matches, the file is
                        kept in good raw data folder for processing. This csv file is missing the first
                        column name,this function changes the missing name to "Wafer".
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        try:
            f = open(self.train_col_valid_log, "a+")

            self.logger.log(f, "Column Length Validation Started!!")

            for file in listdir(self.config["data"]["good"]["train"]):
                csv = pd.read_csv(self.config["data"]["good"]["train"] + "/" + file)

                if csv.shape[1] == NumberofColumns:
                    pass

                else:
                    shutil.move(
                        self.config["data"]["good"]["train"] + "/" + file,
                        self.config["data"]["bad"]["train"],
                    )

                    self.logger.log(
                        f,
                        "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s"
                        % file,
                    )

            self.logger.log(f, "Column Length Validation Completed!!")

        except Exception as e:
            f = open(self.train_col_valid_log, "a+")

            self.logger.log(f, "Error Occured:: %s" % e)

            f.close()

            raise e

    def validateMissingValuesInWholeColumn(self):
        """
        Method Name :   validateMissingValuesInWholeColumn
        Description :   This function validates the misisng values in column in the csv files, and
                        corresponding missing values csv file is created
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        try:
            f = open(self.train_missing_value_log, "a+")

            self.logger.log(f, "Missing Values Validation Started!!")

            for file in listdir(self.config["data"]["good"]["train"]):
                csv = pd.read_csv(self.config["data"]["good"]["train"] + "/" + file)

                count = 0

                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count += 1

                        shutil.move(
                            self.config["data"]["good"]["train"] + "/" + file,
                            self.config["data"]["bad"]["train"],
                        )

                        self.logger.log(
                            f,
                            "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s"
                            % file,
                        )

                        break

                if count == 0:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)

                    csv.to_csv(
                        self.config["data"]["good"]["train"] + "/" + file,
                        index=None,
                        header=True,
                    )

        except Exception as e:
            f = open(self.train_missing_value_log, "a+")

            self.logger.log(f, "Error Occured:: %s" % e)

            f.close()

            raise e
