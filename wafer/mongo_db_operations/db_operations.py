import json

import pandas as pd
from pymongo import MongoClient
from utils.main_utils import read_params


class MongoDBOperation:
    def __init__(self):
        self.config = read_params()

        self.DB_URL = self.config["mongodb"]["url"]

    def get_client(self):
        try:
            self.client = MongoClient(self.DB_URL)

            return self.client

        except Exception as e:
            raise Exception(
                f"Exception occured in Class: MongoDBOperation,Method:get_client, Error: {str(e)}"
            )

    def create_db(self, client, db_name):
        try:
            return client[db_name]

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : MongoDBOperation, Method : create_db, Error : {str(e)}"
            )

    def create_collection(self, database, collection_name):
        try:
            return database[collection_name]

        except Exception as e:
            raise Exception(
                f"Exception occured in Class : MongoDBOperation, Method : create_collection, Error : {str(e)}"
            )

    def get_collection(self, collection_name, database):
        try:
            collection = self.create_collection(database, collection_name)

            return collection

        except Exception as e:
            raise Exception(
                f"Exception occured in class: MongoDBOperation Method : get_collection, Error : {str(e)}"
            )

    def get_collection_as_dataframe(self, db_name, collection_name):
        client = self.get_client()

        database = self.create_db(client, db_name)

        collection = database.get_collection(name=collection_name)

        df = pd.DataFrame(list(collection.find()))

        if "_id" in df.columns.to_list():
            df = df.drop(columns=["_id"], axis=1)

        return df

    def is_record_present(self, db_name, collection_name, record):
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

            raise Exception(
                f"Exception occured in class: MongoDBOperation, Method:is_record_present, Error: {str(e)}"
            )

    def create_one_record(self, collection, data):
        try:
            collection.insert_one(data)

            return 1

        except Exception as e:
            raise Exception(
                f"Exception occured in class: MongoDBOperation, Method:create_one_record, Error: {str(e)} "
            )

    def insert_dataframe_as_record(self, db_name, collection_name, data_frame):
        try:
            records = list(json.loads(data_frame.T.to_json()).values())

            client = self.get_client()

            database = self.create_db(client, db_name)

            collection = self.get_collection(collection_name, database)

            collection.insert_many(records)

            return len(records)

        except Exception as e:
            raise Exception(
                f"Error occured in class: MongoDBOperation, Method:insertDataFrame, Error: {str(e)}"
            )

    def insertRecordInCollection(self, db_name, collection_name, record):
        try:
            client = self.get_client()

            database = self.create_db(client, db_name)

            collection = self.get_collection(collection_name, database)

            if not self.is_record_present(db_name, collection_name, record):
                self.create_one_record(collection=collection, data=record)

            client.close()

        except Exception as e:
            raise Exception(
                "Exception occured in class: MongoDBOperation.\
                    Method:insertRecordInCollection Error:Failed to insert record "
                + str(e)
            )
