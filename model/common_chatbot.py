import os
import operator
import requests
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from typing_extensions import TypedDict
from typing import Annotated,List
from langgraph.graph import START,END,StateGraph

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

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
    notifications_summary: str
    user_query: str
    generation: dict


chat_flow_classifier_template = """You are a Chat Flow Classifier. Based on the provided chat_history and user_query, classify the next response:

- If both chat_history and user_query suggest that a greeting is appropriate, output 'greet'.
- If the user_query suggests that a graph, plot or readings are requested, output 'graph'
- If the user requests for any information regarding a device, batch or to solve any issue, output 'general_qa'.

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

qa_prompt_template = r"""
    You are a helpful question answering assistant.
    Use the chat_history, notifications_summary for generating a response.
    please assist with general inquiries about issues and collect details on issues.
    Don't generate data/information on your self.
    Make the answer simple and concise.

    <Functionality>

    How to Answer General Q/A :-
        - Answer questions related to the user_query

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

    <chat_history>
    {chat_history}
    <\chat_history

    <notifications_summary>
    {notifications_summary}
    <\notifications_summary>

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

qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",qa_prompt_template
            ),
            ("human", "user_query: {user_query}\n notifications_summary:{notifications_summary}"),
        ])

chat_flow_classifier = chat_flow_classifier_prompt | llm | JsonOutputParser()
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

        user_query : {user_query}
        chat_history : {chat_history} """,
            input_variables=["user_query", "chat_history"],
        )

    greeting_bot = prompt | llm | StrOutputParser()
    greeting_bot_response = greeting_bot.invoke({"user_query": user_query, "chat_history": chat_history})

    chat_history = [{"User":user_query, "Bot_Response": greeting_bot_response}]

    generation = {'Type':"text",
                  'response':greeting_bot_response}
    # print(chat_history)


    return {"chat_history": chat_history, "user_query": user_query, "generation" : generation}

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
    notifications_summary = state["notifications_summary"]
    chat_history = state.get("chat_history", [])

    qa_bot_response = qa_bot.invoke({"user_query": user_query, "chat_history":chat_history,"notifications_summary":notifications_summary})

    chat_history = [{"User":user_query, "Bot_Response":qa_bot_response}]
    generation = {'Type':'text',
                  'response':qa_bot_response}

    return {"chat_history":chat_history, "user_query": user_query, "generation": generation}

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

    return state

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    user_query = state["user_query"]
    reference_id = 6 # state["reference_id"]
    # notification = state ["notification"]
    # past_notifications = state["past_notifications"]
    batch_number = 104 # state ["batch_number"]
    # chat_history = state.get("chat_history", [])

    call_back = 0
    filtered_docs = ""

    while call_back < 3:

        print("---RE-WRITTEN QUESTION---")
        print(f"Re-written question: {user_query}")

        # Retrieval
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        vectorstore_humanoid_qa = Chroma(persist_directory=f"docs/chroma/{reference_id}/guidelines_{batch_number}",
                              collection_name=f"guidelines_{batch_number}",
                              embedding_function=embeddings,
                              collection_metadata={"hnsw:space": "cosine"})

        retriever_humanoid_qa = vectorstore_humanoid_qa.as_retriever(search_kwargs={"k":3})

        retrieved_docs = retriever_humanoid_qa.invoke(user_query)

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

    return {"context": filtered_docs, "question": user_query, "chat_history": []}


def common_chatbot():
    workflow = StateGraph(GraphState)

    workflow.add_node("greeting", greeting)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("params_retrieve", params_retrieve)
    workflow.add_node("generate_normal_qa", generate_normal_qa)

    workflow.add_conditional_edges(
        START,
        greeting_classifier,
        {
            "greet": "greeting",
            "graph": "params_retrieve",
            "general_qa": "retrieve",
        },
    )


    workflow.add_edge("retrieve","generate_normal_qa")
    workflow.add_edge("params_retrieve", END)
    workflow.add_edge("generate_normal_qa", END)
    workflow.add_edge("greeting", END)


    graph_model = workflow.compile()

    return graph_model