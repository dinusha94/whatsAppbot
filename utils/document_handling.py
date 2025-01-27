import sys
import os
from sqlalchemy import create_engine,Column,String,Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base_ds = declarative_base()

class DocumentHandlingDB(Base_ds):
    __tablename__="tb_document_handling"

    doc_id = Column("id_value",Integer,primary_key=True,autoincrement=True)
    fileName = Column("fileName",String) 
    filePath = Column("filePath",String)
    vdbName = Column("vdbName",String)

    def __init__(self,fileName,filePath,vdbName):
        self.fileName = fileName
        self.filePath = filePath
        self.vdbName = vdbName

    def __repr__(self):
        return f"({self.doc_id} {self.fileName} {self.filePath} {self.vdbName})"
    
def doc_create_dh_db(connection_string):
    try:
        dh_engine = create_engine(connection_string)
        Base_ds.metadata.create_all(bind=dh_engine)

        return dh_engine
    except Exception as ex:
        print("requested db has already created the hf database - creating only an engine")
        dh_engine = create_engine(connection_string)
        return dh_engine
    
def doc_create_session(engine):
    try:
        Dh_Session = sessionmaker(bind=engine)
        dh_session = Dh_Session()

        return dh_session

    except Exception as ex:
        print("issue with creating session for hf db")
        return None
    
def doc_adding_new_value(session,user_id,source_path):
    try:
        new_feedback = DocumentHandlingDB(user_id=user_id,document_source_path=source_path)
        session.add(new_feedback)
        session.commit()

    except Exception as ex:
        print("Issue in adding new row to the database")

def doc_update_the_value(session,user_id,table_name,col_name,document_source_path):
    try:
        session.query(table_name).filter(table_name.user_id == user_id).update({col_name: document_source_path})
        print("successfully updated")
    except Exception as ex:
        print("There is an issue in updating the selected row")

def doc_db_creation_(connection_string):
    try:
        ds_engine = doc_create_dh_db(connection_string=connection_string)
        ds_session = doc_create_session(ds_engine)
        return ds_session, ds_engine
    except Exception as ex:
        exception = "AN EXCEPTION OCCURRED IN {filename} AND FUNC {method}() AT {line_no}: {ex}".format(
            filename="documents_handling_db.py",
            method="doc_db_creation_",
            line_no=sys.exc_info()[2].tb_lineno,
            ex=ex,
        )
        print("issue in creating db or session: ", exception)
        return None

def document_db():
    doc_db_connection_string = "sqlite:///databases/document_handling.db"
    doc_db_session,doc_db_engine = doc_db_creation_(doc_db_connection_string)

    return doc_db_session,doc_db_engine,doc_db_connection_string