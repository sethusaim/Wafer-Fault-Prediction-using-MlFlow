from datetime import datetime
from src.mongo_db_operations.db_operations import MongoDBOperation


class App_Logger:
    def __init__(self):
        self.db_obj = MongoDBOperation()

    def log(self, db_name, collection_name, log_message):
        try:
            self.now = datetime.now()

            self.date = self.now.date()

            self.current_time = self.now.strftime("%H:%M:%S")

            log = {
                "Log_updated_date": [self.now],
                "Log_updated_time": [self.current_time],
                "Log_message": [log_message],
            }

            self.db_obj.insertRecordInCollection(
                db_name=db_name, collection_name=collection_name, record=log
            )

        except Exception as e:
            raise Exception(
                "Expection occured in Class : App_Logger, Method : log, Error : ",
                str(e),
            )
