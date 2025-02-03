# # from model.common_chatbot import common_chatbot
# # # from graphviz import Digraph
# # from PIL import Image as PILImage

# # common_model = common_chatbot()

# # # graph = common_model.get_graph().draw_png('graph.png')


# # config = {"configurable": {"session_id": 123}}

# # chat_history = []
# # user_query = "what is the meaning of SA_Signalstrenght failier"
# # notifications_summary = "The SAB80SO0071 (Batch: 454_SenzAgro_Valve_Devices) encountered an SA_Signalstrenght failure ."

# # inputs = {
# #     "user_query": user_query,
# #     "chat_history": chat_history,
# #     "notifications_summary": notifications_summary
# # }

# # model_response = common_model.invoke(inputs, config)
# # response = model_response["generation"]["response"]


# # new_chat_entry = {
# #     "Type": "text",
# #     "User": user_query,
# #     "Bot_Response": response
# # }

# # chat_history.append(new_chat_entry)

# from pymongo import MongoClient

# # Connect to the local MongoDB server
# MONGO_URI = "mongodb://localhost:27017/"
# client = MongoClient(MONGO_URI)

# # Reference the database and collection
# db = client["Humanoid"]
# collection = db["NotificationChatHistory"]

# # Insert a sample document into the collection
# sample_data = {"user": "John Doe", "message": "Hello, World!"}
# collection.insert_one(sample_data)

# print("Data inserted successfully!")



import os
from dotenv import load_dotenv
import asyncio
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks, Response

load_dotenv()


# whatsapp credentials
WHAT_TOKEN = os.getenv("ACCESS_TOKEN")   
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PHONE_NUMBER = os.getenv("RECIPIENT_WAID") # my number
VERSION = os.getenv("VERSION")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID") # test number

app = FastAPI()


        
async def handle_incoming_message(request: Request, background_tasks: BackgroundTasks):
    """Handles the incoming message from WhatsApp."""
    try:
        body = await request.json()
        print(body)

        return Response(content="Webhook received successfully!", status_code=200)
    
    except Exception as e:
        return Response(content=f"Error: {str(e)}", status_code=500)

@app.api_route("/webhook", methods=["GET", "POST"])
async def watsapp_bot(request: Request, background_tasks: BackgroundTasks):
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
        return await handle_incoming_message(request, background_tasks)  # Fix: Pass background_tasks


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8007)
