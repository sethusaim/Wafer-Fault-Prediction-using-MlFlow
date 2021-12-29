import re

from utils.logger import App_Logger
from utils.main_utils import convert_object_to_dataframe, raise_exception
from utils.read_params import read_params
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class Raw_Data_validation:
    """
    Description :   This class shall be used for validating the training raw data
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, raw_data_bucket_name):
        self.config = read_params()

        self.raw_data_bucket_name = raw_data_bucket_name

        self.logger = App_Logger()

        self.class_name = self.__class__.__name__

        self.s3_obj = S3_Operations()

        self.good_data_train_bucket = self.config["s3_bucket"]["data_good_train_bucket"]

        self.bad_data_train_bucket = self.config["s3_bucket"]["data_bad_train_bucket"]

        self.db_name = self.config["db_log"]["db_train_log"]

        self.train_schema_log = self.config["train_db_log"]["values_from_schema"]

        self.train_gen_log = self.config["train_db_log"]["general"]

        self.train_name_valid_log = self.config["train_db_log"]["name_validation"]

        self.train_col_valid_log = self.config["train_db_log"]["col_validation"]

        self.train_missing_value_log = self.config["train_db_log"][
            "missing_values_in_col"
        ]

    def values_from_schema(self):
        """
        Method Name :   values_from_schema
        Description :   This method extracts all the relevant information from the pre defined schema file
        Output      :   LengthOfDateStampInFile,LengthOfTimeStampInFile,column_names,NumberofColumns,
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        method_name = self.values_from_schema.__name__

        try:
            dic = self.s3_obj.get_schema_from_s3(
                bucket=self.config["s3_bucket"]["schema_bucket"],
                filename=self.config["schema_file"]["train_schema_file"],
            )

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
                collection_name=self.train_schema_log,
                log_message=message,
            )

        except ValueError:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_schema_log,
                log_message="Exception occured in Class : Raw Data Validation,  \
                    Method : values_from_schema, Error : ValueError:Value not found inside schema_training.json",
            )

            raise ValueError

        except KeyError:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_schema_log,
                log_message="Exception occured in class : Raw Data Validation,  \
                    Method : values_from_schema, Error : KeyError:Key value error incorrect key passed",
            )

            raise KeyError

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=self.db_name,
                collection_name=self.train_schema_log,
            )

        return (
            LengthOfDateStampInFile,
            LengthOfTimeStampInFile,
            column_names,
            NumberofColumns,
        )

    def get_regex_pattern(self):
        """
        Method Name :   get_regex_pattern
        Description :   This method contains a manually defined regex based on the filename given in
                        the schema file
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        method_name = self.get_regex_pattern.__name__

        try:
            regex = "['wafer']+['\_'']+[\d_]+[\d]+\.csv"

            return regex

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=self.db_name,
                collection_name=self.train_gen_log,
            )

    def validate_raw_file_name(
        self, regex, LengthOfDateStampInFile, LengthOfTimeStampInFile
    ):
        """
        Method Name :   validate_raw_file_name
        Description :   This function validates the name of the prediction csv file as per the given name
                        in the schema. Regex pattern is used to do the validation if name format do not match
                        the file is moved to bad raw data folder else in good raw data folder
        On Failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        method_name = self.validate_raw_file_name.__name__

        try:
            onlyfiles = self.s3_obj.get_files_from_s3(
                bucket=self.raw_data_bucket_name,
                db_name=self.db_name,
                collection_name=self.train_name_valid_log,
            )

            for filename in onlyfiles:
                if re.match(regex, filename):
                    splitAtDot = re.split(".csv", filename)

                    splitAtDot = re.split("_", splitAtDot[0])

                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:
                            self.s3_obj.copy_data_to_other_bucket(
                                src_bucket=self.raw_data_bucket_name,
                                src_file=filename,
                                dest_bucket=self.good_data_train_bucket,
                                dest_file=filename,
                                db_name=self.db_name,
                                collection_name=self.train_name_valid_log,
                            )

                        else:
                            self.s3_obj.copy_data_to_other_bucket(
                                src_bucket=self.raw_data_bucket_name,
                                src_file=filename,
                                dest_bucket=self.bad_data_train_bucket,
                                dest_file=filename,
                                db_name=self.db_name,
                                collection_name=self.train_name_valid_log,
                            )

                    else:
                        self.s3_obj.copy_data_to_other_bucket(
                            src_bucket=self.raw_data_bucket_name,
                            src_file=filename,
                            dest_bucket=self.bad_data_train_bucket,
                            dest_file=filename,
                            db_name=self.db_name,
                            collection_name=self.train_name_valid_log,
                        )

                else:
                    self.s3_obj.copy_data_to_other_bucket(
                        src_bucket=self.raw_data_bucket_name,
                        src_file=filename,
                        dest_bucket=self.bad_data_train_bucket,
                        dest_file=filename,
                        db_name=self.db_name,
                        collection_name=self.train_name_valid_log,
                    )

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=self.db_name,
                collection_name=self.train_name_valid_log,
            )

    def validate_col_length(self, NumberofColumns):
        """
        Method Name :   validate_col_length
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
        method_name = self.validate_col_length.__name__

        try:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_col_valid_log,
                log_message="Column Length Validation Started !!",
            )

            csv_file_objs = self.s3_obj.get_file_objs_from_s3(
                bucket=self.good_data_train_bucket,
                db_name=self.db_name,
                collection_name=self.train_col_valid_log,
            )

            for f in csv_file_objs:
                file = f.key

                csv = convert_object_to_dataframe(
                    f, db_name=self.db_name, collection_name=self.train_col_valid_log
                )

                if csv.shape[1] == NumberofColumns:
                    pass

                else:
                    self.s3_obj.move_data_to_other_bucket(
                        src_bucket=self.good_data_train_bucket,
                        src_file=file,
                        dest_bucket=self.bad_data_train_bucket,
                        dest_file=file,
                        db_name=self.db_name,
                        collection_name=self.train_col_valid_log,
                    )

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_col_valid_log,
                log_message="Column Length Validation completed !!",
            )

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=self.db_name,
                collection_name=self.train_col_valid_log,
            )

    def validate_missing_values_in_col(self):
        """
        Method Name :   validate_missing_values_in_col
        Description :   This function validates the misisng values in column in the csv files, and
                        corresponding missing values csv file is created
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        method_name = self.validate_missing_values_in_col.__name__

        try:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_missing_value_log,
                log_message="Missing Values Validation Started!!",
            )

            csv_file_objs = self.s3_obj.get_file_objs_from_s3(
                bucket=self.good_data_train_bucket,
                db_name=self.db_name,
                collection_name=self.train_missing_value_log,
            )

            for f in csv_file_objs:
                file = f.key

                csv = convert_object_to_dataframe(
                    f,
                    db_name=self.db_name,
                    collection_name=self.train_missing_value_log,
                )

                count = 0

                for cols in csv:
                    if (len(csv[cols]) - csv[cols].count()) == len(csv[cols]):
                        count += 1

                        self.s3_obj.move_data_to_other_bucket(
                            src_bucket=self.good_data_train_bucket,
                            src_file=file,
                            dest_bucket=self.bad_data_train_bucket,
                            dest_file=file,
                            db_name=self.db_name,
                            collection_name=self.train_missing_value_log,
                        )

                        break

                if count == 0:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)

                    self.logger.log(
                        db_name=self.db_name,
                        collection_name=self.train_missing_value_log,
                        log_message="Wafer column added to files",
                    )

                    csv.to_csv(file, index=None, header=True)

                    self.logger.log(
                        db_name=self.db_name,
                        collection_name=self.train_missing_value_log,
                        log_message=f"Converted {file} to csv, and created local copy",
                    )

                    self.s3_obj.upload_to_s3(
                        src_file=file,
                        bucket=self.good_data_train_bucket,
                        dest_file=file,
                        db_name=self.db_name,
                        collection_name=self.train_missing_value_log,
                    )

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=self.db_name,
                collection_name=self.train_missing_value_log,
            )
