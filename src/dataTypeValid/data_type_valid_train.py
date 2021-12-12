import csv
import os
import shutil
import sqlite3 as sql_db

from utils.logger import App_Logger
from utils.read_params import read_params


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

        self.db_name = self.config["db_log"]["db_train_log"]

        self.train_db_conn_log = self.config["train_db_log"]["db_conn"]

        self.train_db_create_log = self.config["train_db_log"]["db_create"]

        self.train_db_insert_log = self.config["train_db_log"]["db_insert"]

        self.train_export_csv_log = self.config["train_db_log"]["export_csv"]

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
            db = os.path.join(self.path,DatabaseName,".db")

            conn = sql_db.connect(db)

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_db_conn_log,
                log_message="Opened %s database successfully" % DatabaseName,
            )

        except ConnectionError:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_db_conn_log,
                log_message="Error while connecting to database: %s" % ConnectionError,
            )

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

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_db_create_log,
                    log_message="Tables created successfully!!",
                )

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_db_conn_log,
                    log_message="Closed %s database successfully" % DatabaseName,
                )

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

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_db_create_log,
                    log_message="Tables created successfully!!",
                )

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_db_conn_log,
                    log_message="Closed %s database successfully" % DatabaseName,
                )

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_db_create_log,
                log_message=f"Exception occured in Class : dBOperation, Method : createTableDb, Error : {str(e)}",
            )

            conn.close()

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_db_conn_log,
                log_message="Closed %s database successfully" % DatabaseName,
            )

            raise Exception(
                "Exception occured in Class : dBOperation, Method : createTableDb, Error : ",
                str(e),
            )

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

        onlyfiles = [f for f in os.listdir(goodFilePath)]

        for file in onlyfiles:
            try:
                tmp_file = os.path.join(goodFilePath, file)

                with open(file=tmp_file, mode="r") as f:
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
                                    db_name=self.db_name,
                                    collection_name=self.train_db_insert_log,
                                    log_message=" %s: File loaded successfully!!"
                                    % file,
                                )

                                conn.commit()

                            except Exception as e:
                                raise e

            except Exception as e:
                conn.rollback()

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_db_insert_log,
                    log_message=f"Exception occured in Class : dbOperation. \
                        Method : insertIntoTableGoodData, Error : {str(e)}",
                )

                tmp_file = os.path.join(goodFilePath, file)

                shutil.move(tmp_file, badFilePath)

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.train_db_insert_log,
                    log_message="File Moved Successfully %s" % file,
                )

                conn.close()

        conn.close()

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

            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_export_csv_log,
                log_message="File exported successfully!!!",
            )

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.train_export_csv_log,
                log_message=f"Exception occured in Class : dbOperation, Method : selectingDatafromtableintocsv, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : dbOperation, Method : selectingDatafromtableintocsv, Error : ",
                str(e),
            )
