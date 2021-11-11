from pymongo import MongoClient
from utils.read_params import read_params


class MongoDBOperation:
    def __init__(self):
        self.config = read_params()

        self.DB_URL = self.config["mongodb_url"]

    def getDatabaseClientObject(self):
        try:
            self.client = MongoClient(self.DB_URL)

            return self.client

        except Exception as e:
            raise Exception(
                "Exception occured in Class: MongoDBOperation, \
                    Method:getDataBaseClientObject, Error:Failed to create database connection object-->"
                + str(e)
            )

    def createDatabase(self, client, db_name):
        try:
            return client[db_name]

        except Exception as e:
            raise Exception(
                "Exception occured in MongoDBOperation, method : createDatabase error : "
                + str(e)
            )

    def createCollectionInDatabase(self, database, collection_name):
        try:
            return database[collection_name]

        except Exception as e:
            raise Exception(
                "Exception occured in class : MongoDBOperation method : createCollectionInDatabase error : "
                + str(e)
            )

    def getCollection(self, collection_name, database):
        try:
            collection = self.createCollectionInDatabase(database, collection_name)

            return collection

        except Exception as e:
            raise Exception(
                "Exception occured in class: MongoDBOperation method:getCollection error:Failed to find collection"
                + str(e)
            )

    def isRecordPresent(self, db_name, collection_name, record):

        try:
            client = self.getDatabaseClientObject()

            database = self.createDatabase(client, db_name)

            collection = self.getCollection(collection_name, database)

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
                f"Exception occured in class: MongoDBOperation, Method:isRecordPresent, Error: {str(e)}"
            )

    def createOneRecord(self, collection, data):
        try:
            collection.insert_one(data)

            return 1

        except Exception as e:
            raise Exception(
                "Exception occured in class: MongoDBOperation method:createOneRecord error:Failed to insert record "
                + str(e)
            )

    def insertRecordInCollection(self, db_name, collection_name, record):
        try:
            no_of_row_inserted = 0

            client = self.getDatabaseClientObject()

            database = self.createDatabase(client, db_name)

            collection = self.getCollection(collection_name, database)

            if not self.isRecordPresent(db_name, collection_name, record):
                no_of_row_inserted = self.createOneRecord(
                    collection=collection, data=record
                )

            client.close()

            return no_of_row_inserted

        except Exception as e:
            raise Exception(
                "Exception occured in class: MongoDBOperation.\
                    Method:insertRecordInCollection error:Failed to insert record "
                + str(e)
            )
