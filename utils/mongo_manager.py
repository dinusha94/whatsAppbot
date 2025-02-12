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

    def get_text_content(self, filter_key, message_id):
        _DB = self.client[self.db]
        collection = _DB[self.collection]
        
        query = {filter_key : message_id}
        res = collection.find_one(query, {"message_content": 1, "_id": 0})  # Only retrieve text_content
        
        return res["message_content"] if res else None  # Return text_content if found, else None

    def get_notification_content(self, whatsapp_message_id): 
        _DB = self.client[self.db]
        collection = _DB[self.collection]
        
        query = {"whatsapp_message_id": whatsapp_message_id}
        res = collection.find_one(query, {"message_content": 1, "device_id": 1, "batch_number": 1, "_id": 0})
        
        if res:
            return res.get("message_content"), res.get("device_id"), res.get("batch_number")
        return None, None, None

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

    def check_id_in_list(self, field_name, target_id):
        _DB = self.client[self.db]
        collection = _DB[self.collection]

        res = collection.find_one({}, {field_name: 1, "_id": 0})  

        # Check if the target_id exists in the list
        return target_id in res[field_name] if res and isinstance(res.get(field_name), list) else False



    def append_notification_messages(self, new_entry):
        """
        Append a dictionary (new_entry) to the `notification_messages` list in all documents.

        :param new_entry: A dictionary containing `phone_number` and `message_id`.
        :return: Number of documents modified.
        """
        if isinstance(new_entry, dict):
            _DB = self.client[self.db]
            collection = _DB[self.collection]
            ret = collection.update_many(
                {},  # No filter, applies to all documents
                {"$push": {"notification_messages": new_entry}},  # Append to the lis
                 upsert=True
            )
            return ret.modified_count  # Number of modified documents
        return None
    
    def append_wamid(self, wamid):
        _DB = self.client[self.db]
        collection = _DB[self.collection]
        ret = collection.update_many(
            {},  # No filter, applies to all documents
            {"$push": {"wamid_list": wamid}},  # Append to the lis
                upsert=True
        )
        return ret.modified_count  # Number of modified documents
    
    
    
    
    def update_last_dev_batch(self, new_dev_entry):
        """
        Update or append `new_dev_entry` in the `last_dev_batch` list based on `phone_number`.

        :param new_dev_entry: A dictionary containing `phone_number` and additional keys.
        :return: Number of documents modified.
        """
        if isinstance(new_dev_entry, dict) and "phone_number" in new_dev_entry:
            _DB = self.client[self.db]
            collection = _DB[self.collection]

            # Try updating an existing entry with the same phone_number
            update_result = collection.update_many(
                {"last_dev_batch.phone_number": new_dev_entry["phone_number"]},
                {"$set": {"last_dev_batch.$": new_dev_entry}}
            )

            # If no document was updated, append the new entry instead
            if update_result.modified_count == 0:
                collection.update_many({}, {"$push": {"last_dev_batch": new_dev_entry}})

            return update_result.modified_count  # Number of modified documents

        return None

    def get_last_dev_batch_entry(self, phone_number):
        """
        Retrieve an entry from the `last_dev_batch` list based on the given `phone_number`.

        :param phone_number: The phone number to search for in `last_dev_batch`.
        :return: The matching dictionary or None if not found.
        """
        if not phone_number:
            return None  # Handle invalid input

        _DB = self.client[self.db]
        collection = _DB[self.collection]

        result = collection.find_one(
            {"last_dev_batch.phone_number": phone_number},  # Match document with the phone_number in the list
            {"last_dev_batch.$": 1}  # Use projection to return only the matched entry
        )

        if result and "last_dev_batch" in result:
            return result["last_dev_batch"][0]  # Extract the matching entry

        return None  # Return None if not found


    def get_notification_message_ids_by_phone(self, phone_num):
        """
        Retrieve all notification_message_id values for a given phone number.
        
        :param phone_num: The phone number to filter notifications.
        :return: List of notification_message_id values.
        """
        _DB = self.client[self.db]
        collection = _DB[self.collection]
        
        cursor = collection.find({}, {"notification_messages": 1, "_id": 0})
        
        message_ids = []
        for doc in cursor:
            message_ids.extend([msg.get("notification_message_id") for msg in doc.get("notification_messages", []) if msg.get("phone_num") == phone_num])
        
        return message_ids

    # def insert_user(self, data):
    #     if data is None:
    #         print("Data is None. Insert aborted.")
    #         return None

    #     try:
    #         _DB = self.client[self.db]
    #         collection = _DB[self.collection]

    #         # Check if document already exists based on "name" and "phone_number"
    #         query = {key: data[key] for key in ["name", "phone_number"] if key in data}
    #         existing_doc = collection.find_one(query)

    #         if existing_doc:
    #             print(f"⚠️ Document already exists: {existing_doc['_id']}")
    #             return existing_doc["_id"]  # Return existing document ID instead of inserting

    #         # Insert if not found
    #         ret = collection.insert_one(data)
    #         print(f"Inserted new document ID: {ret.inserted_id}")
    #         return ret.inserted_id

    #     except Exception as e:
    #         print(f"Error inserting document: {e}")
    #         return None

    def insert_user(self, data):
        if data is None:
            print("Data is None. Insert aborted.")
            return None

        try:
            _DB = self.client[self.db]
            collection = _DB[self.collection]

            # Define the query based on phone_number
            query = {"phone_number": data["phone_number"]}

            # Update document if it exists, otherwise insert a new one
            ret = collection.update_one(
                query,  
                {"$set": data},  # Update existing fields with new values
                upsert=True  # Insert if not found
            )

            if ret.matched_count > 0:
                print(f"Updated existing document with phone_number: {data['phone_number']}")
            else:
                print(f"Inserted new document with phone_number: {data['phone_number']}")

            # Retrieve the updated document ID
            updated_doc = collection.find_one(query, {"_id": 1})
            return updated_doc["_id"]

        except Exception as e:
            print(f" Error inserting/updating document: {e}")
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