import csv
import os
import shutil
import sqlite3
from os import listdir

from utils.application_logging.logger import App_Logger
from utils.main_utils import read_params


class dBOperation:
    """
    Description :    This class shall be used for handling all the SQL operations
    Written by  :    iNeuron Intelligence
    Version     :    1.0
    Revisions   :    None
    """

    def __init__(self):
        self.config = read_params()

        self.path = self.config["database_dir"]["train_database"]

        self.badFilePath = self.config["data"]["bad"]["train"]

        self.goodFilePath = self.config["data"]["good"]["train"]

        self.logger = App_Logger()

        self.train_db_conn_log = os.path.join(
            self.config["log_dir"]["train_log_dir"], "DataBaseConnectionLog.txt"
        )

        self.train_db_create_log = os.path.join(
            self.config["log_dir"]["train_log_dir"], "DbTableCreateLog.txt"
        )

        self.train_db_insert_log = os.path.join(
            self.config["log_dir"]["train_log_dir"], "DbInsertLog.txt"
        )

        self.train_export_csv_log = os.path.join(
            self.config["log_dir"]["train_log_dir"], "ExportToCsv.txt"
        )

    def dataBaseConnection(self, DatabaseName):
        """
        Method Name :   dataBaseConnection
        Description :   This method creates the database with the given name and if Database already exists
                        then opens the connection to the db
        Written by  :   iNeuron Intelligence
        Output      :   Connection to the db
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        try:
            conn = sqlite3.connect(self.path + "/" + DatabaseName + ".db")

            file = open(self.train_db_conn_log, "a+")

            self.logger.log(file, "Opened %s database successfully" % DatabaseName)

            file.close()

        except ConnectionError:
            file = open(self.train_db_conn_log, "a+")

            self.logger.log(
                file, "Error while connecting to database: %s" % ConnectionError
            )

            file.close()

            raise ConnectionError

        return conn

    def createTableDb(self, DatabaseName, column_names):
        """
        Method Name :   createTableDb
        Description :   This method creates a table in the given database which will be used to insert the
                        Good data after raw data validation
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        try:
            conn = self.dataBaseConnection(DatabaseName)

            c = conn.cursor()

            c.execute(
                "SELECT count(name)  FROM sqlite_master WHERE type = 'table' AND name = 'Good_Raw_Data'"
            )

            if c.fetchone()[0] == 1:
                conn.close()

                file = open(self.train_db_create_log, "a+")

                self.logger.log(file, "Tables created successfully!!")

                file.close()

                file = open(self.train_db_conn_log, "a+")

                self.logger.log(file, "Closed %s database successfully" % DatabaseName)

                file.close()

            else:
                for key in column_names.keys():
                    type = column_names[key]

                    try:
                        conn.execute(
                            'ALTER TABLE Good_Raw_Data ADD COLUMN "{column_name}" {dataType}'.format(
                                column_name=key, dataType=type
                            )
                        )

                    except:
                        conn.execute(
                            "CREATE TABLE  Good_Raw_Data ({column_name} {dataType})".format(
                                column_name=key, dataType=type
                            )
                        )

                conn.close()

                file = open(self.train_db_create_log, "a+")

                self.logger.log(file, "Tables created successfully!!")

                file.close()

                file = open(self.train_db_conn_log, "a+")

                self.logger.log(file, "Closed %s database successfully" % DatabaseName)

                file.close()

        except Exception as e:
            file = open(self.train_db_create_log, "a+")

            self.logger.log(file, "Error while creating table: %s " % e)

            file.close()

            conn.close()

            file = open(self.train_db_conn_log, "a+")

            self.logger.log(file, "Closed %s database successfully" % DatabaseName)

            file.close()

            raise e

    def insertIntoTableGoodData(self, Database):
        """
        Method Name :   insertIntoTableGoodData
        Description :   This method inserts the Good data files from the Good_raw folder into the above created
                        table
        Written by  :   iNeuron Intelligence
        Versions    :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        conn = self.dataBaseConnection(Database)

        goodFilePath = self.goodFilePath

        badFilePath = self.badFilePath

        onlyfiles = [f for f in listdir(goodFilePath)]

        log_file = open(self.train_db_insert_log, "a+")

        for file in onlyfiles:
            try:
                with open(goodFilePath + "/" + file, "r") as f:
                    next(f)

                    reader = csv.reader(f, delimiter="\n")

                    for line in enumerate(reader):
                        for list_ in line[1]:
                            try:
                                conn.execute(
                                    "INSERT INTO Good_Raw_Data values ({values})".format(
                                        values=(list_)
                                    )
                                )

                                self.logger.log(
                                    log_file, " %s: File loaded successfully!!" % file
                                )

                                conn.commit()

                            except Exception as e:
                                raise e

            except Exception as e:
                conn.rollback()

                self.logger.log(log_file, "Error while creating table: %s " % e)

                shutil.move(goodFilePath + "/" + file, badFilePath)

                self.logger.log(log_file, "File Moved Successfully %s" % file)

                log_file.close()

                conn.close()

        conn.close()

        log_file.close()

    def selectingDatafromtableintocsv(self, Database):
        """
        Method Name :   selectingDatafromtableintocsv
        Description :   This method exports the data in GoodData table as a csv file in a given location
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        self.fileFromDb = self.config["db_file_path"]["train_db_path"]

        self.fileName = self.config["export_csv_file_name"]

        log_file = open(self.train_export_csv_log, "a+")

        try:
            conn = self.dataBaseConnection(Database)

            sqlSelect = "SELECT * FROM Good_Raw_Data"

            cursor = conn.cursor()

            cursor.execute(sqlSelect)

            results = cursor.fetchall()

            headers = [i[0] for i in cursor.description]

            if not os.path.isdir(self.fileFromDb):
                os.makedirs(self.fileFromDb)

            csvFile = csv.writer(
                open(self.fileFromDb + "/" + self.fileName, "w", newline=""),
                delimiter=",",
                lineterminator="\r\n",
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
            )

            csvFile.writerow(headers)

            csvFile.writerows(results)

            self.logger.log(log_file, "File exported successfully!!!")

            log_file.close()

        except Exception as e:
            self.logger.log(log_file, "File exporting failed. Error : %s" % e)

            log_file.close()
