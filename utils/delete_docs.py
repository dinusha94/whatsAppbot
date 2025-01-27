import os
import sys
from sqlalchemy import Column,String,Integer
from sqlalchemy.ext.declarative import declarative_base
from utils.document_handling import doc_create_dh_db,doc_create_session
from utils.create_vdb import load_embedding_model,load_vdb

Base_ds = declarative_base()


class DocumentHandlingDB(Base_ds):
    __tablename__="tb_document_handling"

    doc_id = Column("id_value",Integer,primary_key=True,autoincrement=True)
    fileName = Column("fileName",String) 
    filePath = Column("filePath",String)

    def __init__(self,fileName,filePath,userRole):
        self.fileName = fileName
        self.filePath = filePath

    def __repr__(self):
        return f"({self.doc_id} {self.fileName} {self.filePath})"


def loading_vector_dbs(reference_id,db_name):
    try:

        vdb_path = f"docs/chroma/{reference_id}/guidelines_{db_name}"
        collection_name = f"guidelines_{db_name}"
    
        embedding_model_function = load_embedding_model()
        vdb = load_vdb(db_path=vdb_path, embedding_function=embedding_model_function, index_name=collection_name)

        return vdb

    except Exception as ex:
        exception = "AN EXCEPTION OCCURRED IN {filename} AND FUNC {method}() AT {line_no}: {ex}".format(
            filename="load_vdb.py",
            method="loading_vector_dbs",
            line_no=sys.exc_info()[2].tb_lineno,
            ex=ex,
        )
        raise Exception(exception)
    
def delete_documents(source_list,reference_id,db_name):
    try:
        if type(source_list) is not list:
            source_list = [source_list]    
        
        print(f"source_list: {source_list}")
        
        doc_db_connection_string = "sqlite:///databases/document_handling.db"
        doc_db_engine = doc_create_dh_db(doc_db_connection_string)
        ds_session = doc_create_session(doc_db_engine)

        for single_source in source_list:
            
            ## delete from the database
            doc_delete_db_success = delete_source_from_db(session=ds_session,table_name=DocumentHandlingDB,file_path=single_source)
            
            if not doc_delete_db_success:
                raise ("We could not delete the source file from the db")
            
            vdb = loading_vector_dbs(reference_id,db_name)
            
            source_related_ids = vdb.get(where = {'source': single_source})['ids']
            
            print(f"source related ids: {source_related_ids}")
            
            if len(source_list) > 0:
                vdb.delete(ids=source_related_ids)
            
            print(f"Vector database entries deleted for {single_source}")
            
            os.remove(single_source) ## delete the file from storage
            print(f"Removed the file from storage: {single_source}")

    except Exception as ex:
        exception = "AN EXCEPTION OCCURRED IN {filename} AND FUNC {method}() AT {line_no}: {ex}".format(
            filename="delete_docs.py",
            method="delete_documents",
            line_no=sys.exc_info()[2].tb_lineno,
            ex=ex,
        )
        print(ex)
       
## delete sources from database function - here delete the pdf document -> hence need to delete all the rows
def delete_source_from_db(session,table_name,file_path):
    try:
        session.query(table_name).filter(table_name.filePath == file_path).delete()
        session.commit()
        return True
    except Exception as ex:
        session.rollback()
        print(f"There is an issue in updating the selected row: {ex}")
        return False 