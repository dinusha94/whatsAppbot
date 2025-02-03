__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from glob import glob
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column,String,Integer
from unstructured.cleaners.core import clean_extra_whitespace
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.document_handling import doc_create_dh_db,doc_create_session
import os
from dotenv import load_dotenv
import time
import pandas as pd
import json

load_dotenv()
GOOGLE_API_KEY=os.environ["GOOGLE_API_KEY"]
persist_directory = 'docs/chroma/'

PROCESSING_STATUS_INFO_FILE_PATH = "databases/vdb_status_details.json"

Base_ds = declarative_base()


class DocumentHandlingDB(Base_ds):
    __tablename__="tb_document_handling"

    doc_id = Column("id_value",Integer,primary_key=True,autoincrement=True)
    fileName = Column("fileName",String) ## may be we can change this later to hexadecimal
    filePath = Column("filePath",String)
    vdbName = Column("vdbName",String)

    def __init__(self,fileName,filePath,vdbName):
        self.fileName = fileName
        self.filePath = filePath
        self.vdbName = vdbName

    def __repr__(self):
        return f"({self.doc_id} {self.fileName} {self.filePath} {self.vdbName})"
    
def adding_new_value(session,file_path,vdb_name):
    try:
        file_name = file_path.split("/")[-1]
        new_feedback = DocumentHandlingDB(fileName=file_name,filePath=file_path,vdbName=vdb_name)
        session.add(new_feedback)
        session.commit()
    except Exception as ex:
        print("Issue in adding new row to the database")

def load_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def load_vdb(db_path, embedding_function,index_name):
    vdb = Chroma(persist_directory=db_path,collection_name=index_name,
                 embedding_function=embedding_function)
    print("[INFO] load vector database successfull")
    #################### check for vdb returning
    return vdb

def create_and_save_vdb(texts_docs, embedding_function, db_path, index_name):
    vdb = Chroma.from_documents(texts_docs, embedding=embedding_function,
                                collection_name=index_name, persist_directory=db_path)

    vdb.persist()
    print("[INFO] save the vector dabase")
    return vdb

def load_user_docs_from_db(db_engine,sql_query):
    df_docs = pd.read_sql(sql_query,con = db_engine)
    docs_set = set(df_docs['filePath'])
    return docs_set

def processing_status_update(file_path,update_status = "processing"):
    
    try:
        with open(file_path,'w') as f:
            status_data = {"vector_db_status": update_status}
            # Serializing json
            json_object = json.dumps(status_data, indent=4)
            f.write(json_object)
            
        return True
    except Exception as ex:
        print(f"Error: Issue in writing the vdb status: {ex}")
        return False
    
def process_each_file(item_path,vdb,r_text_splitter,vector_db_path,db_name,embedding_model,ds_session):
    # take extension
    file_extention = item_path.split(".")[-1]
    print(f"file extention {file_extention}")
    
    loader, splitted_docs = None,None
    if file_extention in ["pdf"]:
        loader = UnstructuredFileLoader(item_path,post_processors=[clean_extra_whitespace],)
        splitted_docs = loader.load_and_split(text_splitter=r_text_splitter)
            
        ## Todo: can try to split the documents using recursive splitting method to further split the texts
        
        if len(splitted_docs) > 0:
            
            if vdb is None: ## if vector database is not available create vector database 
                vdb = create_and_save_vdb(texts_docs=splitted_docs,embedding_function=embedding_model,
                    db_path=vector_db_path, index_name=db_name
                    )
            else:
                vdb.add_documents(splitted_docs)
                
            ## add the data sample to database - status is processing
            adding_new_value(ds_session,file_path=item_path,vdb_name=db_name)

def creating_vector_dbs(db_name,reference_id,pdf_dir_path):
    try:

        vector_db_path = f"docs/chroma/{reference_id}/{db_name}"
        doc_db_connection_string = "sqlite:///databases/document_handling.db"

        Path(PROCESSING_STATUS_INFO_FILE_PATH).parent.mkdir(parents=True,exist_ok=True)
        Path(vector_db_path).parent.mkdir(parents=True,exist_ok=True)
        
        if not Path(PROCESSING_STATUS_INFO_FILE_PATH).is_file():
            ## create the file and save it as processing
            vdb_status = processing_status_update(
                PROCESSING_STATUS_INFO_FILE_PATH,
                update_status="idle"
            )

        embedding_model = load_embedding_model()
        r_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

        #########
        doc_db_engine = doc_create_dh_db(doc_db_connection_string)
        ds_session = doc_create_session(doc_db_engine)
        ##############

        vdb = None
        try:
            vdb = load_vdb(vector_db_path, embedding_function=embedding_model, index_name=db_name)
        except Exception as ex_load_vdb:
            vdb = None

        doc_number = 1
        total_processing_time = 0.

        available_docs = load_user_docs_from_db(
            db_engine=doc_db_engine,
            sql_query="select * from tb_document_handling",
        )

        if len(available_docs) > 0:
            pass
        else:
            available_docs = set()
        
        ## save the processing status in the json file
        vdb_status = processing_status_update(
            PROCESSING_STATUS_INFO_FILE_PATH,
            update_status="processing"
        )
        
        if not vdb_status:
            raise ValueError("Error in updating the json file for vdb processing -> processing status...")
        
        ### for all the files in a given directory
        start_time = time.time()

        ### pdf files
        for item_path in glob(pdf_dir_path + "/*.*"):

            if item_path in available_docs:
                continue

            ## extract the texts from document
            doc_number += 1
            print(f"processing doc number: {doc_number}")
            #start_time = time.time()
            process_each_file(
                item_path = item_path,
                vdb = vdb,
                r_text_splitter = r_text_splitter,
                vector_db_path = vector_db_path,
                db_name = db_name,
                embedding_model = embedding_model,
                ds_session = ds_session
            )
        vdb_status = processing_status_update(
            PROCESSING_STATUS_INFO_FILE_PATH,
            update_status="idle"
        )
        
        if not vdb_status:
            raise ValueError("Error in updating the json file for vdb processing -> idle status...")
        
        end_time = time.time()
        total_processing_time = end_time - start_time
        print(f"total processing time: {total_processing_time}")
         
        return vdb


    except Exception as ex:
        exception = "AN EXCEPTION OCCURRED IN {filename} AND FUNC {method}() AT {line_no}: {ex}".format(
            filename="vector_db_genpilot.py",
            method="creating_vector_dbs",
            line_no=sys.exc_info()[2].tb_lineno,
            ex=ex,
        )

        print(exception)

        ## save the processing status in the json file
        vdb_status = processing_status_update(
            PROCESSING_STATUS_INFO_FILE_PATH,
            update_status="idle"
        )

        print(ex)
