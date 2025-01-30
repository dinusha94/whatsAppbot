__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
import shutil
import asyncio
import uvicorn
import base64
import requests
from bson.objectid import ObjectId
from typing import Dict
from datetime import datetime
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient # type: ignore
from fastapi import FastAPI,HTTPException,Request,UploadFile,Form,Depends,Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.notifier_operations import notification_generate,past_notification_generate,rootcause_summary,generate_device_summary
from utils.document_handling import document_db
from utils.create_vdb import creating_vector_dbs
from utils.delete_docs import delete_documents

from utils.whatsapp_utills import parse_whatsapp_message, save_chat

from model.common_chatbot import common_chatbot
from model.models import graph,filter_chat_history,process_chat_history,list_to_string,string_to_list
from pymongo import MongoClient
from langchain_openai import AzureChatOpenAI

from utils.mongo_manager import MongoDBmanager


load_dotenv()
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"] 
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
OPENAI_API_TYPE = os.environ["OPENAI_API_TYPE"]

# whatsapp credentials
WHAT_TOKEN = os.getenv("ACCESS_TOKEN")   
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PHONE_NUMBER = os.getenv("RECIPIENT_WAID") # my number
VERSION = os.getenv("VERSION")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID") # test number


# connect to the mongoDB collections
whatsapp_user_collection = MongoDBmanager("WhatsAppUser")
whatsapp_chat_collection = MongoDBmanager("WhatsAppChat")
whatsapp_message_collection = MongoDBmanager("WhatsAppMessage")
whatsapp_media_collection = MongoDBmanager("WhatsAppMedia")

    
# insert ai agent user
agent_user_id = whatsapp_user_collection.insert_user({
                "name" : "ai_agent",
                "phone_number" : PHONE_NUMBER_ID, 
            })

print(agent_user_id)

llm_config = {"model": "gpt-4o",
             "api_key": AZURE_OPENAI_API_KEY,
             "api_type": OPENAI_API_TYPE,
             "base_url": AZURE_OPENAI_ENDPOINT,
             "api_version": OPENAI_API_VERSION}

llm = AzureChatOpenAI(
    deployment_name="gpt-4o-mini",
    model="gpt-4o-mini"
)

import subprocess

def send_message(response, received_phone_num):
    curl_command = [
        "curl", "-i", "-X", "POST", f"https://graph.facebook.com/{VERSION}/{PHONE_NUMBER_ID}/messages",
        "-H", "Authorization: Bearer EAATgIZBZBKVPIBOw2Y4lIHzYBo9EDg0gjg92sH04bu8AO3EgBk2oFZB4saihZBAeKlHFmblZB9eKUgDB2lmf59dxbDlYZC0iVvgoKczgey38lTPoPsymF4Tg6ZC70tcWuuYvG4XROkpjITwqvxPXrwlF2OeRvEZBC2ZCqtVZA4S87026lhW3ee7EdXZB34SoY8TrKZCTilOZCdmYhnFaTaoZB0ZApZACbfgZBNZCVrL8LtFmUZD",
        "-H", "Content-Type: application/json",
        "-d", f'{{ "messaging_product": "whatsapp", "to": "{received_phone_num}", "type": "text", "text": {{ "body": "{response}" }} }}'
    ]
    
    try:
        # Run the curl command
        result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
        print("Response:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.stderr)


# Initialize FastAPI
app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins = ["*"],
  allow_credentials = True,
  allow_methods = ["*"],
  allow_headers = ["*"]
)

class QueryRequest(BaseModel):
    referenceId: str
    userId: str 
    rootCauseId: str
    question: str
    batchNumber: str

class DeleteFilePayload(BaseModel):
    referenceId: str
    batchNumber: str
    deleteFiles: list[str]

class ConfigUpdateRequest(BaseModel):
    analysis_configs: str
    batchNumber: str
    referenceId: str

class RootCauseRequest(BaseModel):
    referenceId: str
    userId: str 
    rootCauseId: str

class ImgPathRequest(BaseModel):
    img_path: str

# Path to the JSON file
api_config_file = Path("configs/api_configs.json")    ##  configs file need to be created automatically at startup 
analysis_config_file = Path("configs/analysis_configs.json")

# Directory where the uploaded files will be saved
UPLOAD_DIRECTORY = "pdf_chat/"
vdb_path = "docs/chroma/"
vdb_status_details = "databases/vdb_status_details.json"
params_file_path = "databases/params.json"

# OAuth2 Token Validation Configuration
AUTH_SERVER_URL = "https://devicepulse.senzmatica.com/service/validateToken"
AUTH_HEADER_PREFIX = "Bearer"

# Dependency for validating tokens
def validate_token(credentials: HTTPAuthorizationCredentials):
    token = credentials.credentials

    # Validate token using the external auth server
    response = requests.get(
        f"{AUTH_SERVER_URL}?token={token}",
        headers={"Authorization": f"{AUTH_HEADER_PREFIX} {token}"},
    )

    if response.status_code == 200:
        return response.json()
    else:
        message = response.json().get("message", "Token validation failed")
        raise HTTPException(status_code=401, detail=message)

# Reusable security dependency
security = HTTPBearer()

# Example route with token validation
def token_required():
    def wrapper(credentials: HTTPAuthorizationCredentials = Depends(security)):
        return validate_token(credentials)
    return wrapper

@app.post("/get_analysis_configs/")
async def get_analysis_configs(request: Request): # user_data: dict = Depends(token_required()
    # Read the JSON object from the request body
    new_data = await request.json()

    # Extract batchNumber and analysis_configs from the request
    batch_number = new_data.get("batchNumber")
    analysis_configs = new_data.get("analysis_configs")
    reference_id = new_data.get("referenceId")

    # Initialize file_data with an empty structure
    file_data = {"details": {}}

    # Check if the file exists and is not empty
    if os.path.exists(analysis_config_file):
        try:
            with open(analysis_config_file, 'r') as f:
                file_content = f.read().strip()  # Strip whitespace
                if file_content:  # Check if file is not empty
                    file_data = json.loads(file_content)
        except json.JSONDecodeError:
            # Handle case where file contains invalid JSON, re-initialize file_data
            file_data = {"details": {}}
    else:
        # File does not exist, create a new structure
        file_data = {"details": {}}

    # Ensure that 'details' is a dictionary in file_data
    if not isinstance(file_data.get("details"), dict):
        file_data["details"] = {}

    if reference_id not in file_data["details"]:
        file_data["details"][reference_id] = {}

    # Add or update the entry in 'details' using batchNumber as the key
    file_data["details"][reference_id][batch_number] = {
        "analysis_configs": analysis_configs,
        "vectordb": "NULL"
    }

    # Write the updated data back to the file
    with open(analysis_config_file, 'w') as f:
        json.dump(file_data, f, indent=4)

    return {"status": "success"}

@app.put("/edit-analysis-configs")                              # "detail": "Method Not Allowed"
async def edit_analysis_configs(payload: ConfigUpdateRequest):
    # Load the JSON data from the file
    if not analysis_config_file.exists():
        raise HTTPException(status_code=500, detail="Config file not found")
    
    with open(analysis_config_file, "r") as file:
        analysis_config_data = json.load(file)

    batch_number = payload.batchNumber
    reference_id = payload.referenceId

    # Check if the batch number exists in the data
    if reference_id in analysis_config_data["details"]:
        # Update the analysis_configs field only
        analysis_config_data["details"][reference_id][batch_number]["analysis_configs"] = payload.analysis_configs
        
        # Save the updated data back to the file
        with open(analysis_config_file, "w") as file:
            json.dump(analysis_config_data, file, indent=4)

        return {"status": "success", "data": analysis_config_data["details"][reference_id][batch_number]}
    else:
        raise HTTPException(status_code=404, detail="Batch number not found")

#API to show full analysis configs
@app.get("/get_full_analysis_configs/")          # "detail": "Method Not Allowed"                
async def get_full_analysis_configs(request: Request):
    
    # Read the JSON object from the request body
    payload_data = await request.json()

    # Extract batchNumber and analysis_configs from the request
    reference_id = payload_data.get("referenceId")

    # Check if the file exists
    if not os.path.exists(analysis_config_file):
        raise HTTPException(status_code=404, detail="Analysis config file not found")

    try:
        # Read the contents of the JSON file
        with open(analysis_config_file, 'r') as f:
            file_content = f.read().strip()
            if file_content:  # Check if file is not empty
                file_data = json.loads(file_content)
            else:
                file_data = {}  # If file is empty, return an empty object
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding JSON data")
    
    details = file_data.get("details", {})
    reference_details = details.get(reference_id, {})

    return {"details" : reference_details}

"""
{
    "batchNumber": "104",
    "details": {
        "analysis_configs": "Analysis 104",
        "vectordb": "NULL"
    },
    "pdf_files": "EMPTY"
}
"""
#API to show batch details
@app.post("/get_batch_details/")
async def get_batch_details(request: Request):
    # Read the payload from the request body
    request_data = await request.json()
    reference_id = request_data.get("referenceId")
    batch_number = request_data.get("batchNumber")

    # Ensure the batch number is provided
    if not batch_number:
        raise HTTPException(status_code=400, detail="batchNumber is required in the payload")

    # Check if the analysis config file exists
    if not os.path.exists(analysis_config_file):
        raise HTTPException(status_code=404, detail="Analysis config file not found")

    # Load the contents of the analysis config file
    try:
        with open(analysis_config_file, 'r') as f:
            file_content = f.read().strip()
            if not file_content:
                raise HTTPException(status_code=500, detail="Analysis config file is empty")
            file_data = json.loads(file_content)  # Parse the JSON content
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding JSON data")

    # Retrieve details for the given batch number
    batch_details = file_data.get("details", {}).get(reference_id).get(batch_number)
    if not batch_details:
        raise HTTPException(status_code=404, detail=f"No details found for batch number {batch_number}")

    # Construct the path to the PDF directory for the given batch number
    FILE_DIRECTORY = os.path.join(UPLOAD_DIRECTORY,reference_id, batch_number)
    vdb_exist = batch_details["vectordb"]

    # Check if the directory exists
    if not os.path.exists(FILE_DIRECTORY):
        pdf_files = "EMPTY"
    elif vdb_exist == "EXIST":
        pdf_files = {}

        # Iterate over all PDF files in the directory
        pdf_files_paths = [os.path.join(FILE_DIRECTORY, f) for f in os.listdir(FILE_DIRECTORY) if f.endswith('.pdf')]

        for file_path in pdf_files_paths:
            file = Path(file_path)
            with file.open("rb") as f:
                content = f.read()
            
            encoded_content = base64.b64encode(content).decode("utf-8")
            pdf_files[file_path] = encoded_content

    # Return the details and the list of PDF files
    return {
        "batchNumber": batch_number,
        "details": batch_details,
        "pdf_files": pdf_files
    }

"""
seems like checking the PDF files
"""
@app.post("/check-files/")
async def check_files(request: Request):                                 #  "detail": "Directory 'pdf_chat/' does not exist."
    try:
        # Read the JSON object from the request body
        request_data = await request.json()

        # Extract batchNumber and analysis_configs from the request
        reference_id = request_data.get("referenceId")

        if not os.path.exists(UPLOAD_DIRECTORY):
            raise FileNotFoundError(f"Directory '{UPLOAD_DIRECTORY}' does not exist.")

        with analysis_config_file.open("r") as file:
            analysis_data = json.load(file)

        # If reference_id does not exist, initialize it in the JSON file as an empty object
        if reference_id not in analysis_data["details"]:
            analysis_data["details"][reference_id] = {}
            
            # Save the updated JSON back to the file
            with analysis_config_file.open("w") as file:
                json.dump(analysis_data, file, indent=4)
        

        batch_data = analysis_data["details"][reference_id]
        # Check each key and process only if "vectordb" is "EXIST"
        full_batch_data = {}
        for key, value in batch_data.items():
            if value.get("vectordb") == "EXIST":
                subdirectory = os.path.join(UPLOAD_DIRECTORY,reference_id, key)
                if os.path.exists(subdirectory): #and os.path.isdir(subdirectory):
                    files = [f for f in os.listdir(subdirectory) if os.path.isfile(os.path.join(subdirectory, f))]
                    full_batch_data[key] = files
                else:
                    full_batch_data[key] = []  # Subdirectory does not exist or is not a directory
            else:
                full_batch_data[key] = []  # "vectordb" is not "EXIST"

        return full_batch_data
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_file/")
async def upload_file(files: list[UploadFile],json_data: str = Form(...)):
    
    try:
        json_payload = json.loads(json_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data provided")
    
    batch_number = json_payload.get('batchNumber')
    reference_id = json_payload.get("referenceId")

    header_token = json_payload.get('header_token')
    proto = json_payload.get('proto')
        
    # Update the file with new values
    api_configs = {}
    api_configs.update({
        "header_token": header_token,
        "proto": proto
    })

    with open(api_config_file, "w") as api_file:
        json.dump(api_configs, api_file, indent=4)

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    FILE_DIRECTORY = os.path.join(UPLOAD_DIRECTORY,reference_id, batch_number)
    Path(FILE_DIRECTORY).mkdir(parents=True,exist_ok=True)

    with open(analysis_config_file, "r") as file:
        analysis_data = json.load(file)

    saved_files = []
    skipped_files = []
    for file in files:
        if file.content_type != "application/pdf":
            return {"error": "Only PDF files are allowed."}
        
        file_location = os.path.join(FILE_DIRECTORY, file.filename)

        if os.path.exists(file_location):
            skipped_files.append(file.filename)

        else:
            with open(file_location, "wb+") as file_object:
                file_object.write(await file.read())
        
            saved_files.append(file.filename)

    db_file_path = "databases/document_handling.db"
        # Check if the .db file exists
    if os.path.exists(db_file_path):
        print(f"document handling db '{db_file_path}' exists.")
    else:
        print("document handling db doesnt exist")
        doc_db_session,doc_db_engine,doc_db_connection_string = document_db()
        print("document handling db created")

    if batch_number in analysis_data["details"][reference_id]:
        # Update vectordb and add vdb_status field
        analysis_data["details"][reference_id][batch_number]["vdb_status"] = "processing"
    else:
        raise HTTPException(status_code=404, detail=f"Batch number {batch_number} not found in analysis configs")
    
    vectordb_exist = analysis_data["details"][reference_id][batch_number]["vectordb"]
    if vectordb_exist != "EXIST":
        analysis_data["details"][reference_id][batch_number]["vectordb"] = "EXIST"
    
    # Write updated analysis data back to the file
    with open(analysis_config_file, "w") as file:
        json.dump(analysis_data, file, indent=4)
#
    db = creating_vector_dbs(db_name='guidelines'+'_'+batch_number,reference_id = reference_id,pdf_dir_path=FILE_DIRECTORY)
    
    # Update vdb_status to 'idle' after processing files
    analysis_data["details"][reference_id][batch_number]["vdb_status"] = "idle"
    with open(analysis_config_file, "w") as file:
        json.dump(analysis_data, file, indent=4)
    
    result = {
        "info": f"Files saved: {', '.join(saved_files)}" if saved_files else "No new files were saved.",
        "skipped": f"Files skipped (already exist): {', '.join(skipped_files)}" if skipped_files else None
    }
    
    return result

@app.post("/update_file/")
async def update_file(files: list[UploadFile], json_data: str = Form(...)):
    
    try:
        json_payload = json.loads(json_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data provided")
    
    reference_id = json_payload.get('referenceId')
    batch_number = json_payload.get('batchNumber')
    delete_files = json_payload.get('deleteFiles', [])

    if not files and not delete_files:
        raise HTTPException(status_code=400, detail="No files to upload or delete")

    # Directory to save uploaded files based on batch number
    FILE_DIRECTORY = os.path.join(UPLOAD_DIRECTORY,reference_id, batch_number)
    Path(FILE_DIRECTORY).mkdir(parents=True, exist_ok=True)

    # Load existing analysis config data
    with open(analysis_config_file, "r") as file:
        analysis_data = json.load(file)

    saved_files = []
    skipped_files = []
    
    # Delete specified files if they exist
    if delete_files:
        source_list = [os.path.join(FILE_DIRECTORY, f) for f in os.listdir(FILE_DIRECTORY) if f in delete_files]
        delete_documents(source_list=source_list,reference_id = reference_id, db_name=batch_number)
    
    # Check each file and save if it doesn’t already exist
    for file in files:
        if file.content_type != "application/pdf":
            return {"error": "Only PDF files are allowed."}
        
        file_location = os.path.join(FILE_DIRECTORY, file.filename)

        if os.path.exists(file_location):
            skipped_files.append(file.filename)
        else:
            with open(file_location, "wb+") as file_object:
                file_object.write(await file.read())
            saved_files.append(file.filename)

    # Ensure database exists or create it if missing
    db_file_path = "databases/document_handling.db"
    if not os.path.exists(db_file_path):
        doc_db_session, doc_db_engine, doc_db_connection_string = document_db()
        print("Document handling db created.")

    # Update analysis config file status if batch exists
    if batch_number in analysis_data["details"][reference_id]:
        analysis_data["details"][reference_id][batch_number]["vdb_status"] = "processing"
    else:
        raise HTTPException(status_code=404, detail=f"Batch number {batch_number} not found in analysis configs")
    
    # Check if vector database exists
    if analysis_data["details"][reference_id][batch_number].get("vectordb") != "EXIST":
        analysis_data["details"][reference_id][batch_number]["vectordb"] = "EXIST"
    
    # Write updated analysis data back to the file
    with open(analysis_config_file, "w") as file:
        json.dump(analysis_data, file, indent=4)

    # Create or update the vector database with uploaded files
    creating_vector_dbs(db_name='guidelines_' + batch_number,reference_id = reference_id, pdf_dir_path=FILE_DIRECTORY)
    
    # Set status to 'idle' after processing
    analysis_data["details"][reference_id][batch_number]["vdb_status"] = "idle"
    with open(analysis_config_file, "w") as file:
        json.dump(analysis_data, file, indent=4)
    
    result = {
        "info": f"Files saved: {', '.join(saved_files)}" if saved_files else "No new files were saved.",
        "skipped": f"Files skipped (already exist): {', '.join(skipped_files)}" if skipped_files else None,
        "deleted": f"Files deleted: {', '.join(delete_files)}" if delete_files else "No files were deleted."
    }
    
    return result

@app.delete("/delete_configs/")
async def delete_configs(request: Request):
    
    payload = await request.json()
    
    reference_id = payload.get('referenceId')
    batch_number = payload.get("batchNumber")
    FILE_DIRECTORY = os.path.join(UPLOAD_DIRECTORY,reference_id, batch_number)
    VDB_DIRECTORY =  os.path.join(vdb_path,reference_id, f"guidelines_{batch_number}")

    if os.path.exists(FILE_DIRECTORY):
        
        source_list = [os.path.join(FILE_DIRECTORY, f) for f in os.listdir(FILE_DIRECTORY) if f.endswith('.pdf')]
        delete_documents(source_list = source_list,reference_id = reference_id, db_name = batch_number)


        with open(analysis_config_file, "r") as f:
            analysis_configs = json.load(f)
        
        del analysis_configs["details"][reference_id][batch_number]

        with open(analysis_config_file, "w") as f:
            json.dump(analysis_configs, f, indent=4)
        shutil.rmtree(FILE_DIRECTORY)
        shutil.rmtree(VDB_DIRECTORY)    

    else:

        with open(analysis_config_file, "r") as f:
            analysis_configs = json.load(f)
        
        del analysis_configs["details"][reference_id][batch_number]

        with open(analysis_config_file, "w") as f:
            json.dump(analysis_configs, f, indent=4)

    return {"status:success"}

@app.post("/delete_file/")
async def delete_file(payload: DeleteFilePayload):
    
    reference_id = payload.referenceId
    batch_number = payload.batchNumber
    delete_files = payload.deleteFiles

    # Directory path based on batch number
    FILE_DIRECTORY = os.path.join(UPLOAD_DIRECTORY,reference_id, batch_number)
    
    # Verify the directory exists
    if not os.path.exists(FILE_DIRECTORY):
        raise HTTPException(status_code=404, detail=f"Batch directory for batch number {batch_number} not found")
    
    # Initialize list to store file paths that exist
    source_list = []

    # Check each file in the list and verify its existence
    for delete_file in delete_files:
        file_path = os.path.join(FILE_DIRECTORY, delete_file)
        if os.path.exists(file_path):
            source_list.append(file_path)
        else:
            raise HTTPException(status_code=404, detail=f"File '{delete_file}' not found in batch {batch_number}")

     # Delete document entries from the database using source list and batch number
    delete_documents(source_list=source_list,reference_id = reference_id, db_name=batch_number)

    return {"status":"success"}

@app.post("/check_batch_status/")
async def check_batch_status(request: Request):
    
    request_data = await request.json()
    reference_id = request_data.get("referenceId")
    batch_number = request_data.get("batchNumber")

    if not batch_number:
        raise HTTPException(status_code=400, detail="batchNumber is required in the payload.")

    
    with open(analysis_config_file, "r") as file:
        analysis_configs = json.load(file)

    # Check if the batch number exists in the JSON object
    details = analysis_configs.get("details", {})
    reference_data = details.get(reference_id)
    batch_data = reference_data.get(batch_number)

    if batch_number not in reference_data:
        return {"message":"batch not found"}

    # Check the vectordb status
    vectordb_status = batch_data.get("vectordb")

    if vectordb_status == "NULL":
        return {"message": "No vectordb present for this batch."}
    else:
        vdb_status = batch_data.get("vdb_status")
        if vdb_status == "idle":
            return {"message": "No file is processing for this batch."}
        elif vdb_status == "processing":
            return {"message": "File is processing for this batch."}

@app.post("/check_vdb_status/")
async def check_vdb_status(request: Request):

    request_data = await request.json()
    reference_id = request_data.get("referenceId")

    # Initialize `data` as an empty dictionary
    data = {}

     # Check if the file exists and is not empty
    if os.path.exists(vdb_status_details):
        try:
            with open(vdb_status_details, "r") as file:
                file_content = file.read().strip()
                if not file_content:
                    # Empty JSON file
                    status = "sleep"
                data = json.loads(file_content)
        except json.JSONDecodeError:
            # Invalid JSON format
            status  = "sleep"
    else:
        # File does not exist
        status = "sleep"

    # Check if the reference_id exists in the data
    if reference_id not in data:
        status =  "sleep"
    else:
        # Retrieve the status
        status = data[reference_id]["vector_db_status"]


    if status == 'processing':
        return {"Vector DB is creating"}
    elif status == 'idle':
        return {"Vector DB is created"}
    elif status == 'sleep':
        return {"Vector DB is absent"}
    else:
        return {"Error occurred in VectorDB creation"}

@app.post("/save_json/")
async def save_json(request: Request):
    
    rootcause_object = await request.json()
    reference_id = rootcause_object.get("referenceId")
    user_id = rootcause_object.get("userId")
    rootcause_id = rootcause_object.get("rootCauseId")

    session_id = f"{reference_id}_{user_id}_{rootcause_id}"

    MONGO_URI = "mongodb://localhost:27017/"
    client = AsyncIOMotorClient(MONGO_URI)
    db = client["Humanoid"] 
    collection = db["NotificationChatHistory"] 

    existing_document = await collection.find_one({"_id": session_id})
    
    if existing_document:
        
        existing_document = process_chat_history(existing_document)
        return existing_document
    
    else:
        url = f'http://40.76.228.114:9030/core/user/{user_id}/rootCauseDeviceResult/{rootcause_id}'
    
        rootcause_details_response = requests.get(url)
        rootcause_details = rootcause_details_response.json()
        rootcause_content = rootcause_details["content"]
        rootcause_result = rootcause_content["rootCauseResults"][0]["predictions"]
        
        device_id = rootcause_content["deviceId"]
        batch_number = rootcause_content['batchNumber']
        
        notification_task = asyncio.to_thread(notification_generate, llm, rootcause_details)
        past_notifications_task = asyncio.to_thread(past_notification_generate, user_id, rootcause_id, llm, session_id)

        notification, past_notifications = await asyncio.gather(notification_task, past_notifications_task)

        document = {
            "_id": session_id,
            "creationDate":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "referenceId": reference_id,
            "user_id": user_id,
            "rootcause_id": rootcause_id,
            "deviceId" : device_id,
            "batch_number": batch_number,
            "title": rootcause_result,
            "NOTIFICATION": notification,
            "PAST_NOTIFICATIONS": past_notifications,
            "chat_history": []
            }
        
        await collection.insert_one(document)

        return document

@app.get("/list-chat-sessions")
async def list_chat_sessions(request: Request):
    
    request_data = await request.json()
    user_id = request_data.get("userId")
    reference_id = request_data.get("referenceId")

    # Validate required fields
    if not user_id or not reference_id:
        raise HTTPException(status_code=400, detail="Missing 'user_id' or 'reference_id' in request.")

    # MongoDB connection
    MONGO_URI = "mongodb://localhost:27017/"
    client = MongoClient(MONGO_URI)
    db = client["Humanoid"]
    collection = db["NotificationChatHistory"]

    # Query MongoDB for matching documents
    query = {"user_id": user_id, "referenceId": reference_id}
    documents = collection.find(query)

    # Format the response
    response: Dict[str, Dict] = {}
    for doc in documents:
        response[doc["_id"]] = {
            "creationDate": doc.get("creationDate"),
            "deviceId": doc.get("deviceId"),
            "batch_number": doc.get("batch_number"),
            "title": doc.get("title"),
            "notification": doc.get("NOTIFICATION"),
        }

    return response

@app.post("/get-chat-history/")
async def get_chat_history(request: RootCauseRequest):  # use mongo manager class instead

    rootcause_id = request.rootCauseId
    reference_id = request.referenceId
    user_id = request.userId
    session_id = f"{reference_id}_{user_id}_{rootcause_id}"

    MONGO_URI = "mongodb://localhost:27017/"
    client = MongoClient(MONGO_URI)
    db = client["Humanoid"]
    collection = db["NotificationChatHistory"] 

    document = collection.find_one({"_id": session_id})

    raw_chat_history = document["chat_history"]
    notification = document["NOTIFICATION"]

    if isinstance(raw_chat_history, str):
        chat_history = string_to_list(chat_string = raw_chat_history)
    else:
        chat_history = raw_chat_history

    return {"notification" : notification,
            "chat_history" : chat_history}

@app.post("/get-img-content/")
async def get_img_content(request: ImgPathRequest):
    
    img_file_path = request.img_path

    # Check if the file exists
    if not os.path.exists(img_file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Check if the file is a valid JSON file
    if not img_file_path.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only JSON files are allowed.")

    try:
        # Read and parse the JSON file
        with open(img_file_path, "r") as json_file:
            file_content = json.load(json_file)
        return {"status": "success", "data": file_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading JSON file: {str(e)}")

@app.post("/update-acknowledgement/")
async def update_acknowledgement(request: Request):
    
    try:
        request_data = await request.json()
        user_id = request_data.get("userId")
        reference_id = request_data.get("referenceId")
        rootcause_id = request_data.get("rootcauseId")
        acknowledgement_status = request_data.get("Like")

        session_id = f"{reference_id}_{user_id}_{rootcause_id}"

        # MongoDB connection
        MONGO_URI = "mongodb://localhost:27017/"
        client = MongoClient(MONGO_URI)
        db = client["Humanoid"] 
        collection = db["NotificationChatHistory"]

        document = collection.find_one({"_id": session_id})

        if "acknowledgement_status" in document:
            collection.update_one(
                {"_id": session_id}, 
                {"$set": {"acknowledgement_status": acknowledgement_status}}
                )
                
            return {"message": "Acknowledgement field updated successfully"}
        
        else:
            collection.update_one(
                    {"_id": session_id}, 
                    {"$set": {"acknowledgement_status": acknowledgement_status}}
                )
            return {"message": "Acknowledgement field added to the existing document"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/answer_query/")
async def process_query(query: QueryRequest): #### process rootcause quary #######
    
    try:
        question = query.question
        reference_id = query.referenceId
        user_id = query.userId
        batch_number = query.batchNumber
        rootcause_id = query.rootCauseId

        session_id = f"{reference_id}_{user_id}_{rootcause_id}"

        MONGO_URI = "mongodb://localhost:27017/"
        client = MongoClient(MONGO_URI)
        db = client["Humanoid"] 
        collection = db["NotificationChatHistory"] 

        document = collection.find_one({"_id": session_id})
        document = process_chat_history(existing_document = document)
        raw_chat_history = document["chat_history"]
        chat_history = filter_chat_history(raw_chat_history)
        
        device_id = document["deviceId"]
        past_notifications = document["PAST_NOTIFICATIONS"]
        notification = document["NOTIFICATION"]
        user_query = question

        inputs = {"user_query": user_query,
                  "chat_history": chat_history,
                  "device_id": device_id,
                  "batch_number" :batch_number,
                  "notification":notification,                     ## what is this notification is it relewent to the user quary ?
                  "past_notifications":past_notifications,
                  "reference_id":reference_id,
                  "user_id":user_id
                  }
        
        config = {"configurable": {"session_id": session_id}}
        model = graph()

        model_response = model.invoke(inputs,config)

        generation = model_response['generation']
        Type = generation['Type']

        if Type == 'text':
            text_response = generation['response']

            raw_chat_history.append({
                'Type': 'text',
                'User': user_query,
                'Bot_Response': text_response
                })
            
            chat_history_str = list_to_string(chat_list=raw_chat_history)

             # Update the document in MongoDB
            collection.update_one(
                {"_id": session_id},  
                {
                    "$set": {
                        "chat_history": chat_history_str,  
                        "creationDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
                        }
                    }
                )

            return {"result" : text_response}

        else:
            graph_response = generation['response']

            # Generate file path with incremental numbering
            folder_path = os.path.join("model/chat_histories", session_id)
            os.makedirs(folder_path, exist_ok=True)

            # Find the next available number for the file
            file_count = 1
            while os.path.exists(os.path.join(folder_path, f"img_{file_count}.json")):
                file_count += 1
            new_file_path = os.path.join(folder_path, f"img_{file_count}.json")

            # Save the results_data to the new file
            with open(new_file_path, 'w') as new_file:
                json.dump(graph_response, new_file, indent=4)
            print(f"Saved results_data to {new_file_path}")

            raw_chat_history.append({
                'Type': 'graph',
                'User': user_query,
                'Bot_Response': new_file_path
                })

            chat_history_str = list_to_string(chat_list=raw_chat_history)

            # Update the document in MongoDB
            collection.update_one(
                {"_id": session_id},  
                {
                    "$set": {
                        "chat_history": chat_history_str,  
                        "creationDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
                        }
                    }
                )

            # Adding a new key-value pair
            graph_response['file_path'] = new_file_path

            return {"graph_details": graph_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/common_chat/")
async def common_chat(request: Request):

    payload = await request.json()
    session_id = payload.get("sessionId")
    user_id = payload.get("userId")
    reference_id = payload.get("referenceId")
    user_query = payload.get("user_query")

    # MongoDB setup
    client = MongoClient("mongodb://localhost:27017") 
    db = client["Humanoid"]
    collection = db["CommonChatHistory"]

    if session_id:
        # Handle existing session
        try:
            document = collection.find_one({"_id": ObjectId(session_id)})
            if not document:
                raise HTTPException(status_code=404, detail="Session not found")

            # Process and filter the chat history
            document = process_chat_history(document)
            raw_chat_history = document.get("chat_history", [])
            chat_history = filter_chat_history(raw_chat_history)
            notifications_summary = document.get("notifications_summary", "")

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid sessionId: {str(e)}")
    else:
        # Handle new session
        summary_input = rootcause_summary(user_id=user_id)
        notifications_summary = generate_device_summary(llm=llm, summary_input=summary_input)
        chat_history = []

        new_document = {
            "userId": user_id,
            "referenceId": reference_id,
            "notifications_summary": notifications_summary,
            "chat_history": []
        }
        session_id = str(collection.insert_one(new_document).inserted_id)

    # Chatbot interaction
    config = {"configurable": {"session_id": session_id}}
    inputs = {
        "user_query": user_query,
        "chat_history": chat_history,
        "notifications_summary": notifications_summary
    }
    common_model = common_chatbot()
    model_response = common_model.invoke(inputs, config)
    response = model_response["generation"]["response"]

    # Update chat history
    new_chat_entry = {
        "Type": "text",
        "User": user_query,
        "Bot_Response": response
    }
    chat_history.append(new_chat_entry)

    chat_history = list_to_string(chat_list=chat_history)

    collection.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"chat_history": chat_history}}
    )

    return {
        "sessionId": session_id,
        "result": response
    }



@app.api_route("/webhook", methods=["GET", "POST"])
async def watsapp_bot(request: Request):
    if request.method == 'GET':
        # Access query parameters in FastAPI
        mode = request.query_params.get('hub.mode')
        verify_token = request.query_params.get('hub.verify_token')
        challenge = request.query_params.get('hub.challenge')

        print("Mode:", mode)
        print("Verify Token:", verify_token)

        if mode and verify_token:
            if mode == 'subscribe' and verify_token == VERIFY_TOKEN:
                # Return 200 response with the challenge
                return Response(content=challenge, status_code=200)
            else:
                # Return 403 response for invalid verify token
                return Response(content="Forbidden access", status_code=403)
        else:
            # Return 400 response for missing parameters
            return Response(content="No data provided", status_code=400)

    elif request.method == 'POST':
        # Access JSON body in POST request
        body = await request.json()
        # print("POST Body:", body)
        
        parsed_messages = parse_whatsapp_message(body)
        
        # echo the recieved massage
        
        for message in parsed_messages:
            
            # TODO : run the chat bot and send the answer
            response = "llm_response"

            # send the responce to user
            send_message(message['message_content'], message['sender_phone'])
            
            # save the chat history
            save_chat(message, response, agent_user_id, \
                    whatsapp_user_collection, whatsapp_chat_collection, whatsapp_message_collection, whatsapp_media_collection)

        # Process the body (your logic goes here)
        return Response(content="Webhook received successfully!", status_code=200)

   
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8007)
