from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import uvicorn
import logging
import sys
import re
from io import StringIO
import pandas as pd

#print msg in red, accept multiple strings like print statement
def print_red(*strings):
  print('\033[91m' + ' '.join(strings) + '\033[0m')

# print msg in blue, , accept multiple strings like print statement
def print_blue(*strings):
  print('\033[94m' + ' '.join(strings) + '\033[0m')

# Load environment variables from .env file
load_dotenv()

app = FastAPI(debug=True)

client = OpenAI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenAI API key from environment variable
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    raise HTTPException(status_code=500, detail="OpenAI API key is missing")

openai.api_key = api_key

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define request and response models
class QueryRequest(BaseModel):
    prompt: str
    data: list  # This will contain the parsed CSV data from the frontend

class QueryResponse(BaseModel):
    description: str
    chartSpec: dict  # Vega-Lite chart specification

# TOOL 1: Tool to generate Vega-Lite chart from the dataset
def vegaLiteTool(query: str, request: dict):
    # Check if data is provided
    if not request.get("data"):
        return {
            "description": "No dataset uploaded. Please upload a dataset to generate charts.",
            "chartSpec": {}
        }

    try:
        # Gather dataset structure information for GPT-4
        columns = list(request["data"][0].keys())
        column_types = {
            col: "categorical" if isinstance(request["data"][0][col], str) else "quantitative"
            for col in columns
        }
        
        # Construct the prompt for Vega-Lite specification generation
        prompt = f"""
        You are a data visualization assistant. The user provided a dataset with columns: 
        {json.dumps(column_types, indent=2)}.
        
        Here is the full dataset: {json.dumps(request["data"], indent=2)}.
        
        The user has requested visualization with the question: {query}.
        
        Please create a valid Vega-Lite JSON chart specification. Include a 'description' of chart features, analysis, 
        and the type of visualization chosen.

        You will provide the response in this JSON format: 
        {{
            "description": "Description of the chart and its features.",
            "chartSpec": {{
                The valid Vega-Lite JSON chart specification!
            }}
        }}
        """

        # Log the prompt for debugging
        # logger.info(f"Generated prompt: {prompt}")

        # Call OpenAI API to generate chart specification
        gpt_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data visualization assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        # Extract and parse the response
        response_content = gpt_response.choices[0].message.content
        chart_response_json = json.loads(response_content)

        # Extract description and chartSpec from response
        # Extract the description and chartSpec

        description = chart_response_json.get("description", "")
        chart_spec = chart_response_json.get("chartSpec", {})
        
        # Return the extracted data without wrapping in QueryResponse
        return {
            "description": description,
            "chartSpec": chart_spec
        }

    except openai.RateLimitError as e:
        logger.error(f"RateLimitError: {str(e)}")
        raise HTTPException(status_code=429, detail=f"OpenAI API rate limit exceeded.")

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error decoding JSON response.")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error occurred.")


# Function to generate data analysis from the dataset using GPT-4 and pandas 
# Sanitize input to prevent potential security issues
def sanitize_input(query: str) -> str:
    """Sanitize input to the Python REPL by removing unnecessary or potentially unsafe characters."""
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    query = re.sub(r"(\s|`)*$", "", query)
    return query

# Execute Pandas DataFrame code dynamically
def execute_panda_dataframe_code(code: str, dataframe) -> str:
    """
    Execute the provided Python code and return captured output.
    """
    old_stdout = sys.stdout  # Save current stdout to restore later
    sys.stdout = mystdout = StringIO()  # Redirect stdout to capture prints

    try:
        # Sanitize the input code
        cleaned_code = sanitize_input(code)
        # Execute the sanitized code
        exec(cleaned_code)
        # Restore stdout and return captured output
        sys.stdout = old_stdout
        return mystdout.getvalue()
    except Exception as e:
        # Restore stdout and return the error message if an exception occurs
        sys.stdout = old_stdout
        return repr(e)

# TOOL 2: Data analysis tool that receives a code query and a DataFrame
def dataAnalysisTool(query: str, request: dict) -> str:
    """
    Use GPT-4 to generate Python code for analyzing the dataset.
    Execute the generated code and return the result.
    """
    # Check if data is provided
    if not request.get("data"):
        return {
            "description": "No dataset uploaded. Please upload a dataset to generate charts.",
            "chartSpec": {}
        }
    # Gather dataset structure information for GPT-4
    columns = list(request["data"][0].keys())
    column_types = {
        col: "categorical" if isinstance(request["data"][0][col], str) else "quantitative"
        for col in columns
    }

    # Turn request.data in pandas dataframe
    dataframe = pd.DataFrame(request.get("data"))
    # Generate a prompt for the OpenAI API to create the analysis code
    code_prompt = f"""
    You are a data analysis assistant. A user has asked you to: {query}
    The dataset is provided in a variable named 'dataframe'. 
    The dataframe has the following columns: 
        {json.dumps(column_types, indent=2)}.

    Provide Python code using the 'dataframe' variable to perform the analysis, 
    and include print(...) statements to display the output directly.
    You should provide only the python code. 

    You will then be reprompted with the result of the code and will be asked to provide some analysis of the results.
    """


    try:
        # Call OpenAI API to generate analysis code
        gpt_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": code_prompt}
            ],
            temperature=0.3,
        )

        # Extract and parse the response content
        gpt_generated_code = gpt_response.choices[0].message.content

        # Execute the generated code and capture output
        return execute_panda_dataframe_code(gpt_generated_code, dataframe=dataframe)

    except openai.RateLimitError as e:
        logger.error(f"Rate limit error: {e}")
        raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded.")

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=500, detail="Error decoding JSON response.")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred.")

vegaLiteJSON = {
    "name": "vegaLiteTool",
    "description": "Generate a Vega-Lite chart specification based on a dataset and a user prompt. Use this function whenever a user requests a data visualization or chart.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question for the function to answer, well defined and well formatted as a string. Sometimes it may need to be expanded on from the user's original query."
            },
        },
        "required": ["query"],
        "additionalProperties": False
    }
}


# JSON description of dataAnalysisTool
dataAnalysisJSON = {
    "name": "dataAnalysisTool",
    "description": "Generate and execute Python code to analyze a dataset based on a user query. Use this function whenever a user requests a data analysis task.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question for the function to answer, well defined and well formatted as a string. Sometimes it may need to be expanded on from the user's original query."
            },
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

# Aggregate both tools 
tools = [vegaLiteJSON, dataAnalysisJSON]

# Mapping tools to their functions 
tool_map = {
    "vegaLiteTool": vegaLiteTool,
    "dataAnalysisTool": dataAnalysisTool
}

import re
import json

def chat(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

@app.post("/query/", response_model=QueryResponse)
async def query_openai(request: QueryRequest):
    columns = list(request.model_dump()["data"][0].keys())
    column_types = {
        col: "categorical" if isinstance(request.model_dump()["data"][0][col], str) else "quantitative"
        for col in columns
    }

    system_prompt = f'''
    You are a data assistant.

    You have access to this dataset with the following columns:
    {column_types}

    Answer the user question as best you can. You have access to the following tools:
    {tools}

    You run in a loop of Thought, Action, Observation in the following manner to answer the user question.

    Question: the input question you must answer

    Thought: you should always think about what to do
    Action: the tool name, should be one of [{tool_map}]. If no tool need, just output "no tool"
    Action Input: the input to the tool in a json format ({{"arg name": "arg value"}}). Otherwise, empty json object {{}}
    Action Inputs should conform to the valid schema for each tool, both tools require the same schema:
    
    String: query

    You will return this action and action input, then wait for the Observation.

    You will be then call again with the result of the action.

    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Please ALWAYS start with a Thought.

    Please ALWAYS put your response in valid JSON format like so: 
        Thought: 
        Action: 
        Action Input: 
        Final Answer: (When you know the final answer)

    '''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.prompt}
    ]

    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        # Get the assistant's response and add it to conversation history
        response_message = chat(messages)
        print_red(str(response_message))
        messages.append({"role": "assistant", "content": response_message})

        # Use regex to search for Action and Action Input
        match = re.search(r'"Action"\s*:\s*"([^"]+)",\s*"Action Input"\s*:\s*(\{[^}]+\})', response_message)
        if match:
            action_name = match.group(1)
            if action_name == "no tool":
                break  # Stop if no tool is needed

            # Parse action input JSON
            action_input = json.loads(match.group(2))
            # logger.info(f"Model selected tool: {action_name} with arguments: {action_input}")

            # Locate and execute the selected tool
            function_to_call = tool_map.get(action_name)
            action_input["request"] = request.model_dump()

            if function_to_call:
                result = function_to_call(**action_input)
                print_blue(str(result))

                # Format observation as a string
                observation = f"Observation: action name: {action_name}, action_input: {json.dumps(action_input)}, result: {result}"
                messages.append({"role": "assistant", "content": observation})

                # If further interpretation is needed for `dataAnalysisTool`
                if action_name == "dataAnalysisTool":
                    response_json = chat(messages)
                    response_message = json.loads(response_json)  # Convert JSON string to dictionary

                    print_red(str(response_message))

                    # Now access the 'Final Answer' field
                    response_message_content = response_message["Final Answer"]
                
                if action_name == "vegaLiteTool":
                    chartSpec = result.get("chartSpec", {})


                # Prepare response based on tool type
                description = response_message_content if action_name == "dataAnalysisTool" else result.get("description", "")
                chartSpec = chartSpec if action_name == "vegaLiteTool" else {}

                # Return the final structured response
                return QueryResponse(
                    description=description,
                    chartSpec=chartSpec if chartSpec else {}
                )

        # No action identified; return final response
        else:
            return QueryResponse(
                description=response_message,
                chartSpec={}
            )

        iteration += 1

    logger.warning("Max iterations reached without resolving the query.")
    return QueryResponse(
        description="Unable to complete the request within iteration limits.",
        chartSpec={}
    )


# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "API is running"}

# Ensure proper port handling for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
