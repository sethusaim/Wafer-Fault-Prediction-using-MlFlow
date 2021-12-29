from datetime import datetime
from wafer.mongo_db_operations.db_operations import MongoDBOperation


class App_Logger:
    def __init__(self):
        self.db_obj = MongoDBOperation()

        self.class_name = self.__class__.__name__

    def log(self, db_name, collection_name, log_message):
        method_name = self.log.__name__

        try:
            self.now = datetime.now()

            self.date = self.now.date()

            self.current_time = self.now.strftime("%H:%M:%S")

            log = {
                "Log_updated_date": self.now,
                "Log_updated_time": self.current_time,
                "Log_message": log_message,
            }

            self.db_obj.insert_one_record(
                db_name=db_name, collection_name=collection_name, record=log
            )

        except Exception as e:
            error_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(error_msg)
