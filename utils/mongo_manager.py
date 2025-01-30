"""
 * @class MongoDBmanager
 * @description MongoDBmanager use for connect and perfome operation with mongoDB database 
"""

import pathlib
from typing import List, Optional
import pymongo
from pymongo import MongoClient
from pymongo.errors import AutoReconnect, ConnectionFailure
from pymongo.command_cursor import CommandCursor
import bson
import os
from dotenv import load_dotenv
from utils.logger import get_debug_logger

from datetime import datetime
from bson.objectid import ObjectId

from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()  # take environment variables from .env.
# DB_USER = os.environ.get("DB_USER")
# DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DATABASE")
IP = os.environ.get("DB_HOST")
PORT = int(os.environ.get("DB_PORT"))

if not os.path.exists(pathlib.Path.joinpath(pathlib.Path(__file__).parent.resolve(), "../logs")):
    os.makedirs(pathlib.Path.joinpath(pathlib.Path(__file__).parent.resolve(), "../logs"))

logger = get_debug_logger(
    "mongo_manager", pathlib.Path.joinpath(pathlib.Path(__file__).parent.resolve(), "../logs/server.log")
)


class MongoDBmanager:
    def __init__(self, collection, save_json=False):
        self.db = DB_NAME
        self.collection = collection
        
        # Connect to the DB
        try:
            # self.client = MongoClient(f"mongodb://{DB_USER}:{DB_PASS}@{IP}:{PORT}/{DB_NAME}")
            if not save_json:
                self.client = MongoClient(f"mongodb://{IP}:{PORT}/{DB_NAME}")
            else:
                self.client = AsyncIOMotorClient(f"mongodb://{IP}:{PORT}/{DB_NAME}")
            
            logger.debug(f"successfully connected to {DB_NAME} db")
        except (AutoReconnect, ConnectionFailure) as e:
            logger.error(f"failed to connect to {DB_NAME} db, error: {e}")
            raise Exception("DB CONNECTION ERROR")

    """
    get one document by query
    """

    def get_one_document(self, query):
        _DB = self.client[self.db]
        collection = _DB[self.collection]
        res = collection.find_one(query)
        return res

    """
    get documents by query
    """

    def get_documents(self, query):
        _DB = self.client[self.db]
        collection = _DB[self.collection]
        ret = collection.find(query)
        return ret

    def aggregate(self, query) -> Optional[List[dict]]:
        """
        A function to aggregate data using the specified query and return a list of dictionaries or None.
        """
        # Validation
        if query is None or not isinstance(query, list):
            logger.debug("aggregate | N/A | Invalid aggregation query: {}".format(query))
            return None

        _DB = self.client[self.db]
        collection = _DB[self.collection]

        try:
            cursor: CommandCursor = collection.aggregate(query)
            return list(cursor)  # Convert cursor to list
        except pymongo.errors.PyMongoError as e:
            logger.error(f"aggregate | N/A | Aggregation error: {e}")
            return None
        except Exception as e:
            logger.error(f"aggregate | N/A | Unexpected error during aggregation: {e}")
            return None

    """
    insert documents by bulk_write
    """

    def bulk_write(self, query):
        if query != None and len(query) > 0:
            _DB = self.client[self.db]

            collection = _DB[self.collection]
            ret = collection.bulk_write(query, ordered=True)
            return ret
        else:
            logger.debug("No query to bulk_write")

    """
    insert one document by insert_one
    """

    def insert_one(self, data):
        if data != None:
            _DB = self.client[self.db]
            collection = _DB[self.collection]
            ret = collection.insert_one(data)
            return ret

    def insert_user(self, data):
        if data is None:
            print("Data is None. Insert aborted.")
            return None

        try:
            _DB = self.client[self.db]
            collection = _DB[self.collection]

            # Check if document already exists based on "name" and "phone_number"
            query = {key: data[key] for key in ["name", "phone_number"] if key in data}
            existing_doc = collection.find_one(query)

            if existing_doc:
                print(f"⚠️ Document already exists: {existing_doc['_id']}")
                return existing_doc["_id"]  # Return existing document ID instead of inserting

            # Insert if not found
            ret = collection.insert_one(data)
            print(f"Inserted new document ID: {ret.inserted_id}")
            return ret.inserted_id

        except Exception as e:
            print(f"Error inserting document: {e}")
            return None


    

    def insert_chat(self, data):
        if data is None:
            print("Data is None. Insert aborted.")
            return None

        try:
            _DB = self.client[self.db]
            collection = _DB[self.collection]

            # Check if a document with the same user_id already exists
            query = {"user_id": data.get("user_id")}
            existing_doc = collection.find_one(query)

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_message_ids = data.get("message_ids", []) # get current messages

            if existing_doc:
                # Append new message_ids to the existing list
                collection.update_one(
                    query, 
                    {
                        "$set": {"last_updated_timestamp": now}, 
                        "$addToSet": {"message_ids": {"$each": new_message_ids}}  # Append new unique message_ids
                    }
                )
                print(f"⚠️ Document already exists. Updated last_updated_timestamp and appended message_ids: {existing_doc['_id']}")
                return existing_doc["_id"]  # Return existing document ID

            # Prepare the new document
            document = {
                "_id": str(ObjectId()),  # Generate a unique string _id
                "user_id": data["user_id"],
                "start_timestamp": data.get("start_timestamp", now),
                "last_updated_timestamp": now,
                "message_ids": new_message_ids  # Store provided message IDs
            }

            # Insert the document
            ret = collection.insert_one(document)
            print(f"Inserted new document ID: {ret.inserted_id}")
            return ret.inserted_id

        except Exception as e:
            print(f"Error inserting/updating document: {e}")
            return None


    """
    update one document by query
    """

    def find_one_update(self, find_query, update_query, array_filters):
        _DB = self.client[self.db]
        collection = _DB[self.collection]
        try:
            collection.update_one(find_query, {"$set": update_query}, array_filters=array_filters)
        except pymongo.errors.PyMongoError as e:
            logger.error(f"update_one | N/A | update_one error: {e}")
            return Exception("Failed to update the db record")
        except Exception as e:
            logger.error(f"aggregate | N/A | Unexpected error during update_one: {e}")
            return Exception("Failed to update the db record, unexpected error")

    """ 
    find one add to set list 
    """

    def find_one_add_to_set(self, find_query, push_query):
        _DB = self.client[self.db]
        collection = _DB[self.collection]
        try:
            collection.update_one(find_query, {"$addToSet": push_query})
        except bson.errors.InvalidId:
            logger.debug("find_one_add_to_set(): UNSUPPORTED ID")
            raise Exception("UNSUPPORTED_ID")

    """
    update one document by query
    """

    def find_one_update_inc(self, find_query, update_query, array_filters):
        _DB = self.client[self.db]
        collection = _DB[self.collection]
        try:
            collection.update_one(find_query, {"$inc": update_query}, array_filters=array_filters)
        except bson.errors.InvalidId:
            logger.debug("find_one_update_inc(): UNSUPPORTED ID")
            raise Exception("UNSUPPORTED_ID")

    """
    Find one document, add elements to a set field, and update other fields using $set.
    """

    def find_one_add_to_set_and_update(self, find_query, push_query, update_query, array_filters=None):
        _DB = self.client[self.db]
        collection = _DB[self.collection]
        try:
            collection.update_one(
                find_query, {"$addToSet": push_query, "$set": update_query}, array_filters=array_filters
            )
        except bson.errors.InvalidId:
            logger.debug("find_one_add_to_set_and_update(): UNSUPPORTED ID")
            raise Exception("UNSUPPORTED_ID")