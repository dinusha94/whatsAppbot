from model.common_chatbot import common_chatbot
from graphviz import Digraph
from PIL import Image as PILImage

common_model = common_chatbot()

# graph = common_model.get_graph().draw_png('graph.png')


config = {"configurable": {"session_id": 123}}

chat_history = []
user_query = "what is the meaning of SA_Signalstrenght failier"
notifications_summary = "The SAB80SO0071 (Batch: 454_SenzAgro_Valve_Devices) encountered an SA_Signalstrenght failure ."

inputs = {
    "user_query": user_query,
    "chat_history": chat_history,
    "notifications_summary": notifications_summary
}

model_response = common_model.invoke(inputs, config)
response = model_response["generation"]["response"]


new_chat_entry = {
    "Type": "text",
    "User": user_query,
    "Bot_Response": response
}

chat_history.append(new_chat_entry)