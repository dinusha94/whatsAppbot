import operator
import os
import json
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from typing_extensions import TypedDict
from typing import Annotated, List
from pandasai import Agent # type: ignore
from langgraph.graph import START,END,StateGraph
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI


load_dotenv()
GOOGLE_API_KEY=os.environ["GOOGLE_API_KEY"]
OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"] 
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
OPENAI_API_TYPE = os.environ["OPENAI_API_TYPE"]


llm = AzureChatOpenAI(
    deployment_name="gpt-4o-mini",
    model="gpt-4o-mini"
)

class GraphState(TypedDict):

    chat_history: Annotated[List[str], operator.add]
    notification : str
    past_notifications : str
    user_query: str
    context: str
    device_id: str
    batch_number :str
    reference_id: str
    user_id: str
    generation: dict
    api_params : dict
    data_points : dict

question_rewriter_prompt_template = """You are a user_query re-writer that converts an input user_query to a better version that is optimized \n
     for vectorstore retrieval. Look at the user_query  and chat_history ,formulate an improved user_query based on chat_history, past_notifications and notification.
     if it is not need to re-wirte Don't do it let's parse the same input user_query as output.
     \n ------ \n
     notification : {notification}
     \n ------ \n
     past_notifications : {past_notifications}
     \n ------ \n
     chat_history : {chat_history}
     \n ------ \n
     user_query : {user_query}
     \n ------ \n
     Improved question with no preamble: \n """

chat_flow_classifier_template = """You are a Chat Flow Classifier. Based on the provided chat_history and user_query, classify the next response:

- If both chat_history and user_query suggest that a greeting is appropriate, output as 'greet'.
- If the user_query suggests that the user is gratituding, output as 'greet'
- If the user_query suggests that a graph, data points or readings are requested, output as 'graph'
- If the user_query suggests information about issues, past occurrences or troubleshooting steps, output as 'general_qa'.

Facts:
-------
chat_history:
{chat_history}
-------
user_query:
{user_query}
-------

Output either one of 'greet' or 'graph' or 'general_qa' to classify the next response.

Respond with a JSON object containing a single key 'score' and the classification as the value. No additional explanation or preamble."""

qa_classifier_prompt_template = """You are a question Flow Classifier. Based on the provided context,user_query, and chat_history classify the next response:\n

- If the user_query is about a previous issue, specific topic/question and from the context,then output 'normal_qa'.\n
- If the user_query is part of troubleshooting any issue or error, then output as 'troubleshoot'.\n

Following are some example and styles of what could be come under normal_qa,\n

-The device screen is blank. Could it be a power issue?\n
-Why is the solar panel shutting down when it is at lunch time?\n
-What does fault code 404 mean? I see it being displayed in the UI screen.\n
-Can you tell me if this issue has been encountered previously?\n

Following are some example and styles of what could be come under troubleshoot,\n

-I’m getting a notification about low connection status issues and low AC power output. Could any device connection be the cause?\n
-The grid frequency keeps fluctuating. I see fault code 501 is mentioned too. What’s going on?\n

Facts:
-------
chat_history:
{chat_history}
-------
context:
{context}
-------
user_query:
{user_query}
-------

Output either 'normal_qa' or 'troubleshoot' to classify the next response.

Respond with a JSON object containing a single key 'type' and the classification as the value. No additional explanation or preamble.
"""
params_determiner_template = """

You are a Parameter Determiner. Based on the provided user_query and present_time, you must determine the sensor_code, start and end dates:

- You will be provided with present date and time in str data type with the following format Year-month-Day hours:minutes:seconds by present_time.
- If the user_query doesn't mention anything about time provide the start_date and end_date for the last 24 hours
- You must provide start_date, end_date with the same format provided in present_date in str data type.
- start_date will always be behind end_date.

Facts:
-------
present_time:
{present_time}
-------
user_query:
{user_query}
-------

Output sensor_code,start_date, end_date.

Respond with a JSON object containing keys 'sensor_code','start_date','end_date' for the respective values. No additional explanation or preamble."""

query_scope_reducer_template = """
     You are a query re-writer that converts an input user_query to a better version that is optimized for datapoints retrieval.\n
     You must formulate an improved user_query without the inclusion of time,device parameters in any sort.\n
     Do not include about the time or device parameters in the output query.\n
     You must rewrite the question assuming that it will be understood by an agent who has access to a dataframe with the relevant columns of time and value and the agent only requires a query to filter the datapoints based on the conditions.\n
     The following are some examples.\n

    <Example>
    \n

    user_query :- Provide me a plot graph for the days between 24-10-2024 and 24-11-2024\n
    output :- Provide me all the datapoints

    \n

    user_query :- Provide me a plot graph for values greater than 300 for the sensor code AC_E in the last 24 hours\n
    output :- Provide me all the datapoints that are greater than 300

    \n

    user_query :- Provide me a plot graph for values greater than 300 for the sensor code AC_E in the last 24 hours\n
    output :- Provide me all the datapoints that are greater than 300

    \n

    user_query :- Provide me a plot graph for the last 10 data points\n
    output :- Provide me the last 10 data points

    \n
    <\Example>


    if it is not need to re-wirte Don't do it let's parse the same input user_query as output.
    \n ------ \n
    user_query : {user_query}
    \n ------ \n
    Improved question with no preamble: \n """

troubleshoot_prompt_template = r"""
    You are a helpful customer troubleshooting assistant.
    Use the chat_history, context for generating a response.
    please assist in troubleshooting by collecting details about the issues and list down all existing solutions  related with solving the issue from provided context.
    If the context mentions steps that require on-field steps please mention them explicitly and highlight the importance of being in the field. 
    Don't generate data/information on your self. use "context" to get accurate answer.
    If more information is needed for troubleshooting ask the relevent questions. 
    Provide the answers in markdown language.

    <Functionality>

    Steps on General Troubleshooting:-
        - Answer questions related to the products, issues, errors and solutions.
        - list all existing solutions and information when user ask about how to solve a specific issue

    The following is an example conversation you should carry out.

    Example conversation :-
    \n
    User : I am getting the following error. Fault code 101 on the inverter number “0121”. What I have to do\n

    Assistant: Fault code 101 indicates a grid under-voltage issue. This can happen when the voltage supplied by the grid is below the acceptable operating range for the inverter. Let’s go through some troubleshooting steps to resolve this.
            1)Check the grid connection voltage using a multimeter.\n
            2)Ensure the inverter is correctly configured for the grid type (single-phase or three-phase).\n
            3)Verify that the grid voltage matches the inverter’s operating range.\n
    Would you like to start with step 1 and check the grid voltage?\n

    User: Yes, I have a multimeter ready.\n
    Assistant: Great! Here’s how to check the grid voltage:\n
            1)Turn off the inverter to ensure safety.\n
            2)set your multimeter to AC voltage mode.\n
            3)Measure the voltage at the AC terminals of the inverter (where the grid connects).
            4)Compare the measured voltage with the inverter’s specification from the user manual.\n
            Please go ahead and take the measurement. Let me know the voltage reading when you’re ready.\n

    User :- My issues are resolved so far. Thank You\n
    Assistant :- You are welcome. If there's anything let me know.\n



    <\Functionality>

    <Handling false scenarios & Exceptions>
        - Block responses to queries about unknown details.
        - If the user asks a question that is not related to context Politely redirect the user to relevant topics and keep the conversation focused on the intended scope.
            Example: "I’m here to help with questions about troubleshooting devices. Is there something specific you’d like to know?"
        - If the user provides an input that is too vague or unclear for the chatbot to understand. Ask the user to clarify their question, guiding them to rephrase or provide more context.
            Example: "I’m not sure I understand. Could you please provide more details or ask your question in a different way?"
        - If the user inputs a query in a language not supported by the chatbot. Inform the user about the language limitation and request them to switch to a supported language.
            Example: "I’m currently only able to assist in English. Could you please rephrase your question in English?"
        - If the user asks a complex question that requires human intervention or is out of the chatbot’s predefined scope. Offer to escalate the query and collect contact information for a follow-up.
            Example: "This seems like a question for our experts. Would you like me to connect you with a human agent or take down your contact details for follow-up?"
        - If the user inputs a humorous or nonsensical query that does not require a serious response. Respond with a light-hearted acknowledgment while steering the conversation back to relevant topics.
            Example: " I’m here to help with any questions you have about troubleshooting your devices."
        - If the user expresses frustration or negative sentiment towards the chatbot or the website. Show empathy and attempt to address the user’s concerns by gathering more information or escalating the issue.
            Example: "I’m sorry to hear that you’re having trouble. I want to help. Could you please tell me more about the issue you’re facing?"
        - If the user inputs inappropriate or offensive language. Politely address the inappropriate behavior and guide the conversation back to a professional tone.
            Example: "I’m here to assist with your questions about troubleshooting. Let’s keep the conversation respectful."
        - IF the user asks multiple questions in a single input, making it difficult for the chatbot to provide a coherent response. Break down the questions and address them individually to ensure clear communication.
    <\Handling false scenarios & Exceptions>

    <context>
    {context}
    <\context>

    <chat_history>
    {chat_history}
    <\chat_history

    <notification>
    {notification}
    <\notification>

    <user_query>
    {user_query}
    <\user_query>

    """

qa_prompt_template = r"""
    You are a helpful question answering assistant.
    Use the chat_history, context for generating a response.
    Before providing answers you need to understand the nature of the product based on the retrieved documents.
    After understanding the nature of the product, answer the questions as a human tech support on that product (as an example if the document is about inverters, then you need to think and answer like an inverter related tech support). 
    When answering you need to limit answers only to the knowledge that you gained from the retrieved context.
    please assist with general inquiries about issues and collect details on issues.
    Don't generate data/information on your self. use context to get accurate answer.
    Provide solutions from the context.
    When the user requests about past_notifications, then only mention about past_notifications. Other than that don't mention it unnecessarily.
    Provide the answer simple,concise and in a humanoid manner.
    Provide the answer in markdown language.

    <Functionality>

    How to Answer General Q/A :-
        - Answer questions related to the products, issues, errors and solutions based on the context.

    The following is an example conversation you should carry out.

    Example conversation :-
    \n

    User :"What does error 302 mean. I saw it just now."\n
    Assistant :Error 302 indicates high grid voltage, causing the inverter to reduce power for stability. Monitor voltage fluctuations, and if frequent, contact the grid operator.
                
    \n


    <\Functionality>

    <Handling false scenarios & Exceptions>
        - Block responses to queries about unknown details.
        - If the user asks a question that is not related to context Politely redirect the user to relevant topics and keep the conversation focused on the intended scope.
            Example: "I’m here to help with questions about resolving issues in devices. Is there something specific you’d like to know?"
        - If the user provides an input that is too vague or unclear for the chatbot to understand. Ask the user to clarify their question, guiding them to rephrase or provide more context.
            Example: "I’m not sure I understand. Could you please provide more details or ask your question in a different way?"
        - If the user inputs a query in a language not supported by the chatbot. Inform the user about the language limitation and request them to switch to a supported language.
            Example: "I’m currently only able to assist in English. Could you please rephrase your question in English?"
        - If the user asks a complex question that requires human intervention or is out of the chatbot’s predefined scope. Offer to escalate the query and collect contact information for a follow-up.
            Example: "This seems like a question for our experts. Would you like me to connect you with a human agent or take down your contact details for follow-up?"
        - If the user inputs a humorous or nonsensical query that does not require a serious response. Respond with a light-hearted acknowledgment while steering the conversation back to relevant topics.
            Example: " I’m here to help with any questions you have about troubleshooting your devices."
        - If the user expresses frustration or negative sentiment towards the chatbot or the website. Show empathy and attempt to address the user’s concerns by gathering more information or escalating the issue.
            Example: "I’m sorry to hear that you’re having trouble. I want to help. Could you please tell me more about the issue you’re facing?"
        - If the user inputs inappropriate or offensive language. Politely address the inappropriate behavior and guide the conversation back to a professional tone.
            Example: "I’m here to assist with your questions about troubleshooting. Let’s keep the conversation respectful."
        - IF the user asks multiple questions in a single input, making it difficult for the chatbot to provide a coherent response. Break down the questions and address them individually to ensure clear communication.
    <\Handling false scenarios & Exceptions>

    <context>
    {context}
    <\context>

    <chat_history>
    {chat_history}
    <\chat_history

    <notification>
    {notification}
    <\notification>

    <past_notifications>
    {past_notifications}
    <\past_notifications>

    <user_query>
    {user_query}
    <\user_query>

    """

chat_flow_classifier_prompt = ChatPromptTemplate.from_messages(
       [
            (
                "system",chat_flow_classifier_template
            ),
            ("human", "user_query: {user_query}"),
        ])

question_rewriter_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",question_rewriter_prompt_template
            ),
            ("human", "user_query: {user_query}\n notification:{notification}\n past_notifications:{past_notifications}"),
        ])

qa_classifier_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",qa_classifier_prompt_template
            ),
            ("human", "user_query: {user_query}\n context:{context}"),
        ])

troubleshoot_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",troubleshoot_prompt_template
            ),
            ("human", "user_query: {user_query}\n notification:{notification}\n context:{context}"),
        ])

qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",qa_prompt_template
            ),
            ("human", "user_query: {user_query}\n notification:{notification}\n chat_past_notifications:{past_notifications}\n context:{context}"),
        ])

params_determiner_prompt = ChatPromptTemplate.from_messages(
       [
            (
                "system",params_determiner_template
            ),
            ("human", "user_query: {user_query}\n present_time: {present_time}")
       ])

query_rewriter_prompt = ChatPromptTemplate.from_messages(
       [
            (
                "system",query_scope_reducer_template
            ),
            ("human", "user_query: {user_query}")
        ])

question_rewriter = question_rewriter_prompt | llm | StrOutputParser()
chat_flow_classifier = chat_flow_classifier_prompt | llm | JsonOutputParser()
qa_classifier = qa_classifier_prompt | llm | JsonOutputParser()
query_rewriter = query_rewriter_prompt | llm | StrOutputParser()
params_determiner = params_determiner_prompt | llm | JsonOutputParser()
troubleshoot_bot = troubleshoot_prompt | llm | StrOutputParser()
qa_bot = qa_prompt | llm | StrOutputParser()

def greeting(state):
    """
    Provide a greeting response.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New keys added to state, generation and chat_hsitory that contains output, and chat_history.
    """

    print("---GREETING---")


    user_query = state["user_query"]
    chat_history = state.get("chat_history", [])

    # Prompt
    prompt = PromptTemplate(
            template="""
        You are a helpful customer support assistant.
        initiate conversation automatically with a greeting message like "Hi! How can I assist you today?", engaged in a positive tone for interaction.
        provide the the greeting response more politely.
        If the user tries to end the question, greet them with gratitude.
        If the user thanks you for your help, reply them with you're welcome.

        user_query : {user_query}
        chat_history : {chat_history} """,
            input_variables=["user_query", "chat_history"],
        )

    greeting_bot = prompt | llm | StrOutputParser()
    greeting_bot_response = greeting_bot.invoke({"user_query": user_query, "chat_history": chat_history})

    chat_history = [{"User":user_query, "Bot_Response": greeting_bot_response}]
    print(chat_history)

    generation = {'Type':"text",
                  'response':greeting_bot_response}

    return {"chat_history": chat_history, "user_query": user_query, "generation":generation}

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("State", state)

    user_query = state["user_query"]
    reference_id = state["reference_id"]
    notification = state ["notification"]
    past_notifications = state["past_notifications"]
    batch_number = state ["batch_number"]
    chat_history = state.get("chat_history", [])

    call_back = 0
    filtered_docs = ""

    while call_back < 3:

        # Re-Write the questions
        better_question = question_rewriter.invoke({"user_query": user_query, "chat_history":chat_history,"past_notifications":past_notifications, "notification":notification})


        print("---RE-WRITTEN QUESTION---")
        print(f"Re-written question: {better_question}")

        # Retrieval
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # "models/text-embedding-004" is diffrent from the one (models/embedding-001) creating the embeddings

        vectorstore_humanoid_qa = Chroma(persist_directory=f"docs/chroma/{reference_id}/guidelines_{batch_number}",
                              collection_name=f"guidelines_{batch_number}",
                              embedding_function=embeddings,
                              collection_metadata={"hnsw:space": "cosine"})

        retriever_humanoid_qa = vectorstore_humanoid_qa.as_retriever(search_kwargs={"k":3})

        retrieved_docs = retriever_humanoid_qa.invoke(better_question)

        retrieved_docs_str = "\n".join(doc.page_content for doc in retrieved_docs) if retrieved_docs else ""

        if retrieved_docs:
            filtered_docs = retrieved_docs_str
            break
        else:
            call_back += 1

    print("---RETRIEVED DOCUMENTS---")
    #print(f"context: {filtered_docs}")
    print(f"context: {filtered_docs if filtered_docs else 'No relevant documents found.'}")
    for doc in retrieved_docs:
        print(doc.metadata)

    return {"context": filtered_docs, "question": better_question, "chat_history": []}

def generate_troubleshoot(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    print("---GENERATE---")

    user_query = state["user_query"]
    context = state["context"]
    notification =  state["notification"]
    chat_history = state.get("chat_history", [])

    troubleshoot_bot_response = troubleshoot_bot.invoke({"context": context, "user_query": user_query, "chat_history":chat_history,"notification":notification})

    chat_history = [{"User":user_query, "Bot_Response": troubleshoot_bot_response}]

    generation = {'Type':'text',
                  'response':troubleshoot_bot_response}

    return {"chat_history":chat_history, "context": context, "user_query": user_query, "generation": generation}

def generate_normal_qa(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    print("---GENERATE---")

    user_query = state["user_query"]
    context = state["context"]
    past_notifications = state["past_notifications"]
    notification =  state["notification"]
    chat_history = state.get("chat_history", [])

    qa_bot_response = qa_bot.invoke({"context": context, "user_query": user_query, "chat_history":chat_history,"past_notifications":past_notifications,"notification":notification})
    
    chat_history = [{"User":user_query, "Bot_Response": qa_bot_response}]

    generation = {'Type':'text',
                  'response':qa_bot_response}

    return {"chat_history":chat_history, "context": context, "user_query": user_query, "generation": generation}

def greeting_classifier(state):
    """
    Determines whether to generate a greeting.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("State", state)

    print("---CLASSIFY CHAT FLOW---")
    chat_history = state.get("chat_history", [])

    # chat_history = state["chat_history"]
    user_query = state["user_query"]

    # Chat flow classifier
    score = chat_flow_classifier.invoke({"chat_history": chat_history, "user_query": user_query})

    grade = score["score"]

    print("Grade", grade)
    if grade == "greet":
        print("---DECISION: NEXT RESPONSE SHOULD BE GREETING---")
        return "greet"
    elif grade == "graph":
         print("---DECISION: NEXT RESPONSE SHOULD BE PARAMS RETRIEVAL---")
         return "graph"
    else:
      return "general_qa"


def question_classifier(state):
    """
    Determines whether to troubleshoot or qa.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("State", state)

    print("---CLASSIFY CHAT FLOW---")
    chat_history = state.get("chat_history", [])

    # chat_history = state["chat_history"]
    user_query = state["user_query"]
    context = state["context"]

    # Chat flow classifier
    score = qa_classifier.invoke({"chat_history": chat_history, "user_query": user_query,"context":context})

    grade = score["type"]

    print("Grade", grade)
    if grade == "normal_qa":
        print("---DECISION: NEXT STAGE SHOULD BE NORMAL_QA")
        return "normal_qa"
    else:
      return "troubleshoot"

def params_retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, api_params, that contains determined api_params
    """
    print("State", state)
    print("---PARAMETERS RETRIEVE---")

    user_query = state["user_query"]

    now = datetime.now()
    present_time = now.strftime("%Y-%m-%d %H:%M:%S")

    params_object = params_determiner.invoke({"user_query": user_query,"present_time":present_time})

    state["api_params"] = params_object

    return state

def data_retrieval(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, data_point , that contains the data points.
    """
    print("State", state)
    print("---DATA RETRIEVE---")

    params_object = state["api_params"]
    deviceId = state["device_id"]
    user_id = state["user_id"]
    print(f"user_id : {user_id}")

    url = f"http://40.76.228.114:9030/core/user/{user_id}/sensorsByDeviceAndCode"
    params = {
        "deviceId": deviceId,
        "sensorCode": params_object['sensor_code'],
        "from": params_object['start_date'],
        "to": params_object['end_date']
        }

    response = requests.get(url, params=params)
    data = response.json()

    response = requests.get(url, params=params)
    data = response.json()
    print(data)

    state["data_points"] = data

    return state

def data_generation(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state,generation, that contains thenecessary data points from the retrieved data_points
    """
    print("State", state)
    print("---DATA GENERATION---")

    data_points = state["data_points"]
    user_query = state["user_query"]
    deviceId = state["device_id"]
    params_object = state["api_params"]
    sensor_code =  params_object['sensor_code']

    dataset = data_points["content"]
    df = pd.DataFrame(dataset)
    df['value'] = df['value'].astype(float)
    df = df[['time','code','value']]

    rewritten_query = query_rewriter.invoke({"user_query": user_query})
    agent = pandasai_agent(df)
    response = agent.chat(rewritten_query)
    response_df = response[['time','value']]

    # Prepare JSON object
    result = {
        "x_label": "Creation Date",
        "y_label": "Value",
        "title": f"Values vs Creation Date for Device ID: {deviceId}, Code: {sensor_code}",
        "values": response_df['value'].tolist(),
        "creation_dates": response_df['time'].tolist()  # formatted as string
    }

    generation = {'Type' : 'graph',
                  'response' : result}

    state["generation"] = generation

    return state


def graph(): 
    workflow = StateGraph(GraphState)

    workflow.add_node("greeting", greeting)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_troubleshoot", generate_troubleshoot)
    workflow.add_node("generate_normal_qa", generate_normal_qa)
    workflow.add_node("params_retrieve", params_retrieve)
    workflow.add_node("data_retrieval", data_retrieval)
    workflow.add_node("data_generation", data_generation)

    workflow.add_conditional_edges(
        START,
        greeting_classifier,
        {
            "greet": "greeting",
            "graph": "params_retrieve",
            "general_qa": "retrieve",
        },
    )

    workflow.add_conditional_edges(
        'retrieve',
        question_classifier,
        {
            "troubleshoot": "generate_troubleshoot",
            "normal_qa": "generate_normal_qa",
        },
    )

    workflow.add_edge("params_retrieve", "data_retrieval")
    workflow.add_edge("data_retrieval", "data_generation")
    workflow.add_edge("data_generation", END)
    workflow.add_edge("generate_normal_qa", END)
    workflow.add_edge("generate_troubleshoot", END)
    workflow.add_edge("greeting", END)


    graph_model = workflow.compile()

    return graph_model
    
def string_to_list(chat_string):
    try:
        # Convert the JSON string back to a list
        return json.loads(chat_string)
    except Exception as e:
        raise ValueError(f"Error while converting string to list: {str(e)}")
    
def list_to_string(chat_list):
    try:
        # Convert the list to a JSON string
        return json.dumps(chat_list)
    except Exception as e:
        raise ValueError(f"Error while converting list to string: {str(e)}")

def filter_chat_history(chat_history):
    # Check if the list is empty
    if not chat_history:
        return []

    # Filter dictionaries where the 'type' key is equal to 'text'
    filtered_chat_history = [d.copy() for d in chat_history if d.get('Type') == 'text']

    # Remove the 'type' field from each dictionary in the filtered list
    for d in filtered_chat_history:
        d.pop('Type', None)
    
    return filtered_chat_history

def process_chat_history(existing_document):

    chat_history = existing_document.get("chat_history")

    # Check if chat_history is of type list or string
    if isinstance(chat_history, str):
        # If it's a string, convert it to a list and update the JSON object
        try:
            converted_chat_history = string_to_list(chat_history)
            print(converted_chat_history)
            existing_document["chat_history"] = converted_chat_history
        except Exception as e:
            # Handle any issues with string_to_list conversion
            raise ValueError(f"Error converting chat_history to list: {e}")
    elif not isinstance(chat_history, list):
        existing_document["chat_history"] = []  # Ensure it's a list if not already

    return existing_document

def pandasai_agent(df):
  llm = AzureChatOpenAI(
    deployment_name="gpt-4o-mini",
    model="gpt-4o-mini")

  agent = Agent(df, config = {'llm': llm})

  return agent