import re

from utils.logger import App_Logger
from utils.read_params import read_params
from wafer.s3_bucket_operations.s3_operations import S3_Operation


class Raw_Pred_Data_Validation:
    """
    Description :   This method is used for validating the raw Prediction data
    Written by  :   iNeuron Intelligence
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self, raw_data_bucket_name):
        self.config = read_params()

        self.raw_data_bucket_name = raw_data_bucket_name

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

        self.s3 = S3_Operation()

        self.pred_data_bucket = self.config["s3_bucket"]["wafer_pred_data"]

        self.input_files_bucket = self.config["s3_bucket"]["input_files"]

        self.raw_pred_data_dir = self.config["data"]["raw_data"]["pred_batch"]

        self.pred_schema_file = self.config["schema_file"]["pred"]

        self.regex_file = self.config["regex_file"]

        self.pred_schema_log = self.config["pred_db_log"]["values_from_schema"]

        self.good_pred_data_dir = self.config["data"]["pred"]["good"]

        self.bad_pred_data_dir = self.config["data"]["pred"]["bad"]

        self.pred_gen_log = self.config["pred_db_log"]["general"]

        self.pred_name_valid_log = self.config["pred_db_log"]["name_validation"]

        self.pred_col_valid_log = self.config["pred_db_log"]["col_validation"]

        self.pred_missing_value_log = self.config["pred_db_log"][
            "missing_values_in_col"
        ]

    def values_from_schema(self):
        """
        Method Name :   values_from_schema
        Description :   This method gets schema values from the schema_prediction.json file

        Output      :   Schema values are extracted from the schema_prediction.json file
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.values_from_schema.__name__

        try:
            self.log_writer.start_log(
                key="start",
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.pred_schema_log,
            )

            dic = self.s3.read_json(
                fname=self.pred_schema_file,
                bucket_name=self.input_files_bucket,
                log_file=self.pred_schema_log,
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

            self.log_writer.log(
                log_file=self.pred_schema_log, log_info=message,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.pred_schema_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.pred_schema_log,
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
        Description :   This method gets regex pattern from input files s3 bucket

        Output      :   A regex pattern is extracted
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_regex_pattern.__name__

        try:
            self.log_writer.start_log(
                key="start",
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.pred_gen_log,
            )

            regex = self.s3.read_text(
                fname=self.regex_file,
                bucket_name=self.input_files_bucket,
                log_file=self.pred_gen_log,
            )

            self.log_writer.log(
                log_file=self.pred_gen_log, log_info=f"Got {regex} pattern",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.pred_gen_log,
            )

            return regex

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.pred_gen_log,
            )

    def create_dirs_for_good_bad_data(self, log_file):
        """
        Method Name :   create_dirs_for_good_bad_data
        Description :   This method creates folders for good and bad data in s3 bucket

        Output      :   Good and bad folders are created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.create_dirs_for_good_bad_data.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            log_file=log_file,
        )

        try:
            self.s3.create_folder(
                folder_name=self.good_pred_data_dir,
                bucket_name=self.pred_data_bucket,
                log_file=log_file,
            )

            self.s3.create_folder(
                folder_name=self.bad_pred_data_dir,
                bucket_name=self.pred_data_bucket,
                log_file=log_file,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                log_file=log_file,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                log_file=log_file,
            )

    def validate_raw_file_name(
        self, regex, LengthOfDateStampInFile, LengthOfTimeStampInFile
    ):
        """
        Method Name :   validate_raw_file_name
        Description :   This method validates the raw file name based on regex pattern and schema values

        Output      :   Raw file names are validated, good file names are stored in good data folder and rest is stored in bad data
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.validate_raw_file_name.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            log_file=self.pred_name_valid_log,
        )

        try:
            self.create_dirs_for_good_bad_data(log_file=self.pred_name_valid_log)

            onlyfiles = self.s3.get_files_from_folder(
                bucket=self.raw_data_bucket_name,
                folder_name=self.raw_pred_data_dir,
                log_file=self.pred_name_valid_log,
            )

            pred_batch_files = [f.split("/")[1] for f in onlyfiles]

            self.log_writer.log(
                log_file=self.pred_name_valid_log,
                log_info="Got Prediction files with absolute file name",
            )

            for fname in pred_batch_files:
                raw_data_pred_file_name = self.raw_pred_data_dir + "/" + fname

                good_data_pred_file_name = self.good_pred_data_dir + "/" + fname

                bad_data_pred_file_name = self.bad_pred_data_dir + "/" + fname

                self.log_writer.log(
                    log_file=self.pred_name_valid_log,
                    log_info="Created raw,good and bad data file name",
                )

                if re.match(regex, fname):
                    splitAtDot = re.split(".csv", fname)

                    splitAtDot = re.split("_", splitAtDot[0])

                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:
                            self.s3.copy_data(
                                from_fname=raw_data_pred_file_name,
                                from_bucket=self.pred_data_bucket,
                                to_fname=good_data_pred_file_name,
                                to_bucket=self.pred_data_bucket,
                                log_file=self.pred_name_valid_log,
                            )

                        else:
                            self.s3.copy_data(
                                from_fname=raw_data_pred_file_name,
                                from_bucket=self.pred_data_bucket,
                                to_fname=bad_data_pred_file_name,
                                to_bucket=self.pred_data_bucket,
                                log_file=self.pred_name_valid_log,
                            )

                    else:
                        self.s3.copy_data(
                            from_fname=raw_data_pred_file_name,
                            from_bucket=self.pred_data_bucket,
                            to_fname=bad_data_pred_file_name,
                            to_bucket=self.pred_data_bucket,
                            log_file=self.pred_name_valid_log,
                        )
                else:
                    self.s3.copy_data(
                        from_fname=raw_data_pred_file_name,
                        from_bucket=self.pred_data_bucket,
                        to_fname=bad_data_pred_file_name,
                        to_bucket=self.pred_data_bucket,
                        log_file=self.pred_name_valid_log,
                    )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.pred_name_valid_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.pred_name_valid_log,
            )

    def validate_col_length(self, NumberofColumns):
        """
        Method Name :   validate_col_length
        Description :   This method validates the column length based on number of columns as mentioned in schema values

        Output      :   The files' columns length are validated and good data is stored in good data folder and rest is stored in bad data folder
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.validate_col_length.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            log_file=self.pred_col_valid_log,
        )

        try:
            lst = self.s3.read_csv_from_folder(
                folder_name=self.good_pred_data_dir,
                bucket_name=self.pred_data_bucket,
                log_file=self.pred_col_valid_log,
            )

            for idx, f in enumerate(lst):
                df = f[idx][0]

                file = f[idx][1]

                abs_f = f[idx][2]

                if file.endswith(".csv"):
                    if df.shape[1] == NumberofColumns:
                        pass

                    else:
                        dest_f = self.bad_pred_data_dir + "/" + abs_f

                        self.s3.move_data(
                            from_fname=file,
                            from_bucket=self.pred_data_bucket,
                            to_fname=dest_f,
                            to_bucket=self.pred_data_bucket,
                            log_file=self.pred_col_valid_log,
                        )

                else:
                    pass

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.pred_col_valid_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.pred_col_valid_log,
            )

    def validate_missing_values_in_col(self):
        """
        Method Name :   validate_missing_values_in_col
        Description :   This method validates the missing values in columns

        Output      :   Missing columns are validated, and good data is stored in good data folder and rest is to stored in bad data folder
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.validate_missing_values_in_col.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            log_file=self.pred_missing_value_log,
        )

        try:
            lst = self.s3.read_csv_from_folder(
                folder_name=self.good_pred_data_dir,
                bucket_name=self.pred_data_bucket,
                log_file=self.pred_missing_value_log,
            )

            for idx, f in lst:
                df = f[idx][0]

                file = f[idx][1]

                abs_f = f[idx][2]

                if abs_f.endswith(".csv"):
                    count = 0

                    for cols in df:
                        if (len(df[cols]) - df[cols].count()) == len(df[cols]):
                            count += 1

                            dest_f = self.bad_pred_data_dir + "/" + abs_f

                            self.s3.move_data(
                                from_fname=file,
                                from_bucket=self.pred_data_bucket,
                                to_fname=dest_f,
                                to_bucket=self.pred_data_bucket,
                                log_file=self.pred_missing_value_log,
                            )

                            break

                    if count == 0:
                        dest_f = self.good_pred_data_dir + "/" + abs_f

                        self.s3.upload_df_as_csv(
                            data_frame=df,
                            local_file_name=abs_f,
                            bucket_file_name=dest_f,
                            bucket_name=self.pred_data_bucket,
                            log_file=self.pred_missing_value_log,
                        )

                else:
                    pass

                self.log_writer.start_log(
                    key="exit",
                    class_name=self.class_name,
                    method_name=method_name,
                    log_file=self.pred_missing_value_log,
                )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.pred_missing_value_log,
            )
