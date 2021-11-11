import json
import os
import re
import shutil
from datetime import datetime

import pandas as pd
from utils.logger import App_Logger
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

        self.db_name = self.config["db_log"]["db_train_log"]

        self.train_values_from_schema_log = self.config["train_db_log"][
            "values_from_schema"
        ]

        self.train_gen_log = self.config["train_db_log"]["general"]

        self.train_name_valid_log = self.config["train_db_log"]["name_validation"]

        self.train_col_valid_log = self.config["train_db_log"]["col_validation"]

        self.train_missing_value_log = self.config["train_db_log"][
            "missing_values_in_col"
        ]

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
            with open(file=self.schema_path, mode="r") as f:
                dic = json.load(f)

            LengthOfDateStampInFile = dic["LengthOfDateStampInFile"]

            LengthOfTimeStampInFile = dic["LengthOfTimeStampInFile"]

            column_names = dic["ColName"]

            NumberofColumns = dic["NumberofColumns"]

            message = (
                "LengthOfDateStampInFile:: %s" % LengthOfDateStampInFile
                + "\t"
                + "LengthOfTimeStampInFile:: %s" % LengthOfTimeStampInFile
                + "\t "
                + "NumberofColumns:: %s" % NumberofColumns
                + "\n"
            )

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_values_from_schema_log,
                log_message=message,
            )

            f.close()

        except ValueError:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_values_from_schema_log,
                log_message="Exception occured in Class : Raw Data Validation,  \
                    Method : valuesfromschema, Error : ValueError:Value not found inside schema_training.json",
            )

            f.close()

            raise ValueError

        except KeyError:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_values_from_schema_log,
                log_message="Exception occured in class : Raw Data Validation,  \
                    Method : valuesfromschema, Error : KeyError:Key value error incorrect key passed",
            )

            raise KeyError

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_values_from_schema_log,
                log_message=f"Exception occured in Class : Raw Data Validation, \
                     Method : valuesfromSchema, Error : {str(e)}",
            )

            raise Exception(
                f"Exception occured in Class : Raw Data Validation,Method : valuesfromSchema, Error : {str(e)}"
            )

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
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_gen_log,
                log_message=f"Exception occured in Class : Raw_data_validation, \
                    Method :manualRegexCreation, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : Raw_data_validation,Method :manualRegexCreation, Error :",
                str(e),
            )

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

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_gen_log,
                log_message=f"Exception occured in Class : Raw_data_validation, \
                    Method : createDirectoryForGoodBadRawData, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : Raw_data_validation, \
                    Method : createDirectoryForGoodBadRawData, Error : ",
                str(e),
            )

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

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_gen_log,
                    log_message="GoodRaw directory deleted successfully!!!",
                )

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_gen_log,
                log_message=f"Exception occured in Class : Raw_data_validation, \
                    Method : deleteExistingGoodDataTrainingFolder, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : Raw_data_validation, \
                    Method : deleteExistingGoodDataTrainingFolder, Error : ",
                str(e),
            )

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

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_gen_log,
                    log_message="BadRaw directory deleted before starting validation!!!",
                )

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_gen_log,
                log_message=f"Exception occured in Class : Raw_data_validation, \
                    Method : deleteExistingBadDataTrainingFolder, Exception occured : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : Raw_data_validation, \
                    Method : deleteExistingBadDataTrainingFolder, Exception occured : ",
                str(e),
            )

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

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_gen_log,
                    log_message="Bad files moved to archive",
                )

                path = self.config["data"]["bad"]["train"]

                if os.path.isdir(path):
                    shutil.rmtree(path)

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_gen_log,
                    log_message="Bad Raw Data Folder Deleted successfully!!",
                )

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_gen_log,
                log_message=f"Exception occured in class : Raw_data_validation \
                    Method : moveBadFilesToArchiveBad, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in class : Raw_data_validation,, Method : moveBadFilesToArchiveBad, Error :",
                str(e),
            )

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

        onlyfiles = [f for f in os.listdir(self.Batch_Directory)]

        try:
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
                                db_name=self.db_name,
                                collection_name=self.train_name_valid_log,
                                log_message="Valid File name!! File moved to GoodRaw Folder :: %s"
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
                                db_name=self.db_name,
                                collection_name=self.train_name_valid_log,
                                log_message="Invalid File Name!! File moved to Bad Raw Folder :: %s"
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
                            db_name=self.db_name,
                            collection_name=self.train_name_valid_log,
                            log_message="Invalid File Name!! File moved to Bad Raw Folder :: %s"
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
                        db_name=self.db_name,
                        collection_name=self.train_name_valid_log,
                        log_message="Invalid File Name!! File moved to Bad Raw Folder :: %s"
                        % filename,
                    )

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_name_valid_log,
                log_message=f"Exception occured in Class : Raw_data_validation \
                    Method : validationFileNameRaw, Error : {str(e)} ",
            )

            raise Exception(
                "Exception occured in Class : Raw_data_validation \
                    Method : validationFileNameRaw, Error : ",
                str(e),
            )

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
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_col_valid_log,
                log_message="Column Length Validation Started!!",
            )

            for file in os.listdir(self.config["data"]["good"]["train"]):
                csv = pd.read_csv(self.config["data"]["good"]["train"] + "/" + file)

                if csv.shape[1] == NumberofColumns:
                    pass

                else:
                    shutil.move(
                        self.config["data"]["good"]["train"] + "/" + file,
                        self.config["data"]["bad"]["train"],
                    )

                    self.logger.log(
                        db_name=self.db_name,
                        collection_name=self.train_col_valid_log,
                        log_message="Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s"
                        % file,
                    )

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_col_valid_log,
                log_message="Column Length Validation Completed!!",
            )

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_col_valid_log,
                log_message=f"Exception occured in Class : Raw_data_validation, \
                    Method : validateColumnLength, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : Raw_data_validation, Method : validateColumnLength, Error :",
                str(e),
            )

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
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_missing_value_log,
                log_message="Missing Values Validation Started!!",
            )

            for file in os.listdir(self.config["data"]["good"]["train"]):
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
                            db_name=self.db_name,
                            collection_name=self.train_missing_value_log,
                            log_message="Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s"
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
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_missing_value_log,
                log_message=f"Exception occured in class Raw_data_validation, \
                    Method : validateMissingValuesInWholeColumn, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in class Raw_data_validation, \
                    Method : validateMissingValuesInWholeColumn, Error : ",
                str(e),
            )
