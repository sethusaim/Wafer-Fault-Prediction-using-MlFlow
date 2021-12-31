import json

import pandas as pd
from pymongo import MongoClient
from utils.read_params import read_params


class MongoDBOperation:
    def __init__(self):
        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.DB_URL = self.config["mongodb"]["url"]

    def get_client(self):
        method_name = self.get_client.__name__

        try:
            self.client = MongoClient(self.DB_URL)

            return self.client

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def create_db(self, client, db_name):
        method_name = self.create_db.__name__

        try:
            return client[db_name]

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def create_collection(self, database, collection_name):
        method_name = self.create_collection.__name__

        try:
            return database[collection_name]

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def get_collection(self, collection_name, database):
        method_name = self.get_collection.__name__

        try:
            collection = self.create_collection(database, collection_name)

            return collection

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def convert_collection_to_dataframe(self, db_name, collection_name):
        method_name = self.convert_collection_to_dataframe.__name__

        try:
            client = self.get_client()

            database = self.create_db(client, db_name)

            collection = database.get_collection(name=collection_name)

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            return df

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def is_record_present(self, db_name, collection_name, record):
        method_name = self.is_record_present.__name__

        try:
            client = self.get_client()

            database = self.create_db(client, db_name)

            collection = self.get_collection(collection_name, database)

            recordfound = collection.find(record)

            if recordfound.count() > 0:
                client.close()

                return True

            else:
                client.close()

                return False

        except Exception as e:
            client.close()

            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def create_one_record(self, collection, data):
        method_name = self.create_one_record.__name__

        try:
            collection.insert_one(data)

            return 1

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def insert_dataframe_as_record(self, data_frame, db_name, collection_name):
        method_name = self.insert_dataframe_as_record.__name__

        try:
            records = json.loads(data_frame.T.to_json()).values()

            client = self.get_client()

            database = self.create_db(client, db_name)

            collection = database.get_collection(collection_name)

            collection.insert_many(records)

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)

    def insert_one_record(self, db_name, collection_name, record):
        method_name = self.insert_one_record.__name__

        try:
            client = self.get_client()

            database = self.create_db(client, db_name)

            collection = self.get_collection(collection_name, database)

            if not self.is_record_present(db_name, collection_name, record):
                self.create_one_record(collection=collection, data=record)

            client.close()

        except Exception as e:
            exception_msg = f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}"

            raise Exception(exception_msg)
