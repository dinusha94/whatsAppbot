import json
import os
import requests
from langchain_core.output_parsers import StrOutputParser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from typing import List, Dict

def replace_null_true(json_obj):
    if isinstance(json_obj, dict):
        return {k: replace_null_true(v) for k, v in json_obj.items()}
    elif isinstance(json_obj, list):
        return [replace_null_true(item) for item in json_obj]
    elif json_obj is None:  # replace null with None
        return None
    elif json_obj is True:  # replace true with True
        return True
    elif json_obj is False:  # replace false with False if needed
        return False
    return json_obj

def adjusting_function(dictionary, test_case_sensor_vals):
    for k, v in test_case_sensor_vals.items():
        test_result = dictionary[k]
        # If test_result is a string, we keep it as is under the 'result' key
        if isinstance(test_result, str):
            dictionary[k] = {"result": test_result, "code": v['sensorCode']}
        else:
            dictionary[k] = {"result": test_result['result'], "code": v['sensorCode']}
    return dictionary

def get_main_testId(user_id,rootCauseId):

    rootcause_url = f'http://40.76.228.114:9030/core/user/{user_id}/rootCauseDeviceResult/{rootCauseId}'
    response = requests.get(rootcause_url)
    
    rootcause_data = response.json()

    rootcause_content = rootcause_data["content"]
    device_id = rootcause_content["deviceId"]
    mainTestId = rootcause_content['mainTestId']

    return mainTestId,device_id

def get_past_noti_data(user_id,rootCauseId):

    mainTestId,device_id =  get_main_testId(user_id,rootCauseId)
    
    # Step 1 & 2: Get and calculate dates
    now = datetime.now()
    end_date = now #+ timedelta(hours=5, minutes=30) 
    start_date = now - timedelta(hours=8)#, minutes=30)
    start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")
    
    past_noti_url = f'http://40.76.228.114:9030/core/user/{user_id}/rootCauseDetailsByDeviceId?mainTestId={mainTestId}&deviceId={device_id}&from={start_date_str}&to={end_date_str}'
    
    past_noti_response = requests.get(past_noti_url)
    past_noti_obj = past_noti_response.json()

    return past_noti_obj

def process_past_notification(content):
    
    time = content["time"]
    predictions = content["rootCauseResults"][0]["predictions"]
    failed_sub_test_case_name = content["failedSubTestCaseName"]

    return (
        f"Time: {time}\n"
        f"Prediction: {predictions}\n"
        f"Failed Test Case: {failed_sub_test_case_name}\n\n"
    )

def past_notification_generate(user_id,rootCauseId,llm,session_id):
    
    past_notifications = get_past_noti_data(user_id,rootCauseId)
    # Define the prompt template for summarizing incident data
    past_notification_preprocessing_prompt_template = PromptTemplate(
        input_variables=["incident_details"],
        template=(
            """Provide a description for the following incidents in natural language:\n
            Don't include any additional information. Describe each incident by one sentence.\n
            {incident_details}\n
            if there are no incidents in incident_details, reply with no such incidents encountered.
            Provide a bullet point description for each of the incidents:"""
        )
    )

    # Create a LangChain chain with the template and LLM
    past_notification_preprocessing_chain = past_notification_preprocessing_prompt_template | llm

    # Prepare the combined incident details for a single prompt
    incident_details = ""

    # Process notifications concurrently for speed-up
    with ThreadPoolExecutor() as executor:
        processed_notifications = executor.map(process_past_notification, past_notifications["content"])

    # Efficient string concatenation
    incident_details = "".join(processed_notifications)
    summary = past_notification_preprocessing_chain.invoke({"incident_details": incident_details.strip()},session_id=session_id)
    
    return summary.content

def generate_subTest_summary(test_details: List[Dict], llm) -> str:
    # Construct a prompt with test details formatted for clarity
    prompt = """
    Summarize the following test details for a non-technical audience, highlighting sensor codes, test titles, and value ranges. Mention success criteria where available, and make the language simple and concise.
    You will be provided a sub test title, sensor code, and a value range and a test criteria that the device failed. Make it a short summary. 
    Test Details:
    """
    
    # Add each test detail to the prompt in a clear format
    for test in test_details:
        prompt += (
            f"\n- Subtest Title: {test.get('subTestTitle', 'N/A')}\n"
            f"  - Sensor Code: {test.get('sensorCode', 'N/A')}\n"
            f"  - Value Range: {test.get('minVal', 'N/A')} to {test.get('maxVal', 'N/A')}\n"
            f"  - Test Criteria: {test.get('successCriteria', 'N/A')}\n"
        )
    
    # Send the prompt to the LLM for processing
    response = llm.invoke(prompt)
    
    # Extract and return the summary text
    return response.content.strip()

def test_reports_preprocessing(llm,test_results):
    results = test_results['results']
    test_case_info = test_results['testCaseInfo']["subTestCases"]

    # Filter test_case_info to extract relevant info (sensorCode)
    filtered_test_case_info = {item['subTestTitle']: {"sensorCode": item['sensorCode']} for item in test_case_info}

    # Adjust the test results based on the test_case_info
    adjusted_test_results = [adjusting_function(item, filtered_test_case_info) for item in results]

    # Convert results and test case info to JSON strings
    adjusted_test_results_string = json.dumps(adjusted_test_results)
    test_case_info_string = json.dumps(test_case_info)

    # Error summary prompt with placeholders
    error_summary_prompt = """
    You will be provided the test details for different devices and sensors. Specifically device related test details and relevant test criteria related details with sensor code.
    Note that 'code' and 'sensorCode' are the same.
    These are given in json format. You must analyse the given test details and test criteria details and rearrange them in a proper natural language that 'non tech persons' can understand. 
    All important details must be there, including the sensor code (should mention explicitly) and test criteria details if available for each device. Remember to include sensor code appropriately.
    provide concise answers for each device with test criteria details. give the final answer in an explanotary manner for each device.

    Test Details: {error_details}
    Test Criteria Details: {test_criteria_details}
    """

    # Format the prompt with actual test details
    error_summary_prompt_ = error_summary_prompt.format(error_details=adjusted_test_results_string, test_criteria_details=test_case_info_string)

    # Invoke the LLM (adjust according to your LLM usage)
    error_summary_natural_language = llm.invoke(error_summary_prompt_)

    return error_summary_natural_language.content

def notification_generate(llm,rootcause_details):
    
    rootcause_content = rootcause_details["content"]
    time = rootcause_content["time"]
    device_id = rootcause_content["deviceId"]
    batch_number = rootcause_content["batchNumber"]
    rootcause_result = rootcause_content["rootCauseResults"][0]["predictions"]
    failed_subTest_cases = rootcause_content['failedSubTestCase']
    failed_subTest_cases_summary = generate_subTest_summary(test_details = failed_subTest_cases, llm = llm)
    
    notification_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                 """You are a helpful notification assistant.
                You will be provided with the {time}, {device_id}, {batch_number},{failed_subTest_cases_summary} and {rootcause_result}.
                {failed_subTest_cases_summary} include a summary of the sub test cases that were failed.
                {rootcause_result} includes the reason why the sub test cases were failed.
                You should provide a short summary based on the above provided details.
                Do not provide any other unnecessary details. Make it no more than 5 sentences also. Include all provided details.
                Start by saying "Hi you have a notification" and present the summary in a new line.

                \n

                
                <TIME>
                {time}
                </RESOURCES>

                <BATCH NUMBER> 
                {batch_number}
                </BATCH NUMBER>

                <DEVICE ID> 
                {device_id}
                </DEVICE ID>
                
                <FAILED SUBTEST CASES SUMMARY> 
                {failed_subTest_cases_summary}
                </FAILED SUBTEST CASES SUMMARY>

                <ROOTCAUSE RESULT> 
                {rootcause_result}
                </ROOTCAUSE RESULT>
                
                """
                ,
            ),
            ("human", "time: {time}\n device_id : {device_id}\n batch_number: {batch_number}\n failed_subTest_cases_summary: {failed_subTest_cases_summary}\n rootcause_result: {rootcause_result} "),
        ])

    notification_chain = notification_prompt | llm 

    notification_msg= notification_chain.invoke({
        "time": time,
        "device_id": device_id,
        "batch_number":batch_number,
        "failed_subTest_cases_summary": failed_subTest_cases_summary,
        "rootcause_result":rootcause_result
        })
    
    notification = notification_msg.content
    
    return notification

def rootcause_summary(user_id):
    # Define the URL
    url = f"https://devicepulse.senzmatica.com/service/core/user/{user_id}/allRootCauseResults"

    # Define query parameters
    params = {
        "startingIndex": 0,
        "endIndex": 50
    }

    # Make the GET request
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an exception for HTTP errors
    data = response.json()

        # Extract content
    content = data.get('content', [])

    if not content:
      return "No content available for the provided user ID."

    # Extract relevant fields
    extracted_data = [
            {
                "batch_number": item['batchNumber'],
                "device_id": item["deviceId"],
                "failed_sub_test_case_name": item["failedSubTestCaseName"],
                "predictions": item["rootCauseResults"][0]["predictions"],
                "time": item["time"]
            }
            for item in content
        ]

    # Prepare the summary
    summary_input = "\n".join([
            f"Device ID: {data['device_id']}, Failed Sub-Test Case: {data['failed_sub_test_case_name']}, "
            f"Prediction: {data['predictions']}, Time: {data['time']}, Batch Number: {data['batch_number']}"
            for data in extracted_data
        ])

    return summary_input

def generate_device_summary(llm, summary_input):

    summary_prompt_template = '''Here is the extracted data for analysis grouped by Device ID:\n\n{summary_input}\n\n
    Generate a concise summary of the overall findings for each device.
    Within the concise summary, you must include all the information for each device.\n
    The following is an example:\n

    **Device ID: 123456**
    Device 123456 encountered an issue in Sub-Test Case 1 at 11:07:09 on January 8, 2025, during batch 704. The predicted cause of the failure is: prediction.
    \n

    If there are multiple failed times for a single device, mention those within the failed time for that device.
    '''

    # Create the prompt template
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", summary_prompt_template),
            ("human", "summary_input:{summary_input}"),
        ]
    )

    # Create the bot
    summary_bot = summary_prompt | llm | StrOutputParser()

    # Get the summary bot response
    summary_bot_response = summary_bot.invoke({"summary_input": summary_input})

    return summary_bot_response
