# import datetime
from datetime import datetime

def parse_whatsapp_message(data):
    # Validate that the input has the required structure
    if not data.get('entry'):
        raise ValueError("Invalid JSON: 'entry' key not found")
    
    parsed_messages = []
    
    # Loop through each entry
    for entry in data['entry']:
        changes = entry.get('changes', [])
        for change in changes:
            value = change.get('value', {})
            messages = value.get('messages', [])
            metadata = value.get('metadata', {})
            contacts = value.get('contacts', [])
            
            for message in messages:
                # Extract sender's info
                contact = contacts[0] if contacts else {}
                sender_name = contact.get('profile', {}).get('name', "Unknown")
                sender_phone = contact.get('wa_id', "Unknown")
                
                # Extract message details
                message_content = message.get('text', {}).get('body', "No text")
                message_timestamp = message.get('timestamp', "Unknown")
                
                # Convert timestamp (if present) to human-readable format
                if message_timestamp != "Unknown":
                    message_timestamp = datetime.fromtimestamp(
                        int(message_timestamp)
                    ).strftime('%Y-%m-%d %H:%M:%S')
                
                # Collect parsed message info
                parsed_messages.append({
                    "sender_name": sender_name,
                    "sender_phone": sender_phone,
                    "message_content": message_content,
                    "message_timestamp": message_timestamp
                })
    
    return parsed_messages

def save_chat(user_message, responce, agent_user_id, user_collection, chat_collection, message_collection, media_collection):
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    question = user_message['message_content']
    question_timestamp = user_message['message_timestamp']
    phone_number = user_message['sender_phone']
    sender_name = user_message['sender_name']
   
    # create user if not previously created
    user_id = user_collection.insert_user({
                "name" : sender_name,
                "phone_number" : phone_number,
            })
    
    # insert user question
    user_question_id = message_collection.insert_one(
        {
            "timestamp" : question_timestamp,
            "sender_id" : user_id,
            "recipient_id" : agent_user_id,
            "message_type" : "text",
            "media_id" : "",
            "message_content" : question
        
        })
    
    # insert llm responce
    llm_response_id = message_collection.insert_one(
        {
            "timestamp" : current_time,
            "sender_id" : agent_user_id,
            "recipient_id" : user_id,  
            "message_type" : "text",
            "media_id" : "",
            "message_content" : responce
            
        })
    
    # create a chat
    chat_id = chat_collection.insert_chat(
                {
                    "user_id": user_id,
                    "start_timestamp": current_time,
                    "last_updated_timestamp": current_time,
                    "message_ids" : [user_question_id.inserted_id, llm_response_id.inserted_id]
                })
    
    
    
    
    
    
    
    
