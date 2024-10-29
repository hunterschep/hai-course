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

# Helper functions for printing model thought process 
def print_red(*strings):
  print('\033[91m' + ' '.join(strings) + '\033[0m')

def print_blue(*strings):
  print('\033[94m' + ' '.join(strings) + '\033[0m')

# Load environment variables from .env file
load_dotenv()

app = FastAPI(debug=True)
client = OpenAI()

# initialize core_data as an empty DataFrame
core_data = pd.DataFrame()

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
    table: str # markdown code for table 
    chartSpec: dict  # Vega-Lite chart specification

# TOOL 1: Tool to generate Vega-Lite chart from the dataset
def vegaLiteTool(query: str, request: dict):
    global core_data
        
    # Increase display settings for debugging
    pd.set_option('display.max_rows', None)  # or use a high number like 500
    pd.set_option('display.max_columns', None)  # Ensure all columns are displayed
    pd.set_option('display.width', 1000)  # Increase the width of each row

    # Check if data is provided
    if not request.get("data"):
        return {
            "description": "No dataset uploaded. Please upload a dataset to generate charts.",
            "chartSpec": {}
        }

    try:
        # Construct the prompt for Vega-Lite specification generation
        prompt = f"""
        You are a data visualization assistant. The user provided this data with these columns: {core_data.columns}.
        
        Here is the dataset: {core_data}.
        
        The user has requested visualization with the question: {query}.
        
        Please create a valid Vega-Lite JSON chart specification. Include a 'description' of chart features, analysis, 
        and the type of visualization chosen.

        You will provide the response in this JSON format: 
        {{
            "description": "Description of the chart with some complementary analysis.",
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

# TOOL 1 - Helper function: Clean the input to be ran as code
def sanitize_input(query: str) -> str:
    """Sanitize input to the Python REPL by removing unnecessary or potentially unsafe characters."""
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    query = re.sub(r"(\s|`)*$", "", query)
    return query

# TOOL 1 - Helper function: Run the python code 
def execute_panda_dataframe_code(code: str, dataframe) -> tuple:
    """
    Execute the provided Python code and return both captured output and the modified DataFrame.
    """
    old_stdout = sys.stdout  # Save current stdout to restore later
    sys.stdout = mystdout = StringIO()  # Redirect stdout to capture prints

    try:
        # Sanitize the input code
        cleaned_code = sanitize_input(code)
        # Create a local dictionary to serve as the local namespace for execution
        local_vars = {'dataframe': dataframe}
        # Execute the sanitized code with the DataFrame in the local namespace
        exec(cleaned_code, globals(), local_vars)
        # Capture the possibly modified DataFrame
        modified_dataframe = local_vars['dataframe']
        # Restore stdout
        sys.stdout = old_stdout
        # Return the captured output and the modified DataFrame
        return mystdout.getvalue(), modified_dataframe
    except Exception as e:
        # Restore stdout and return the error message if an exception occurs
        sys.stdout = old_stdout
        return repr(e), dataframe

# TOOL 2: Data analysis tool that receives a code query and a DataFrame
def dataAnalysisTool(query: str, request: dict) -> str:
    global core_data 
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
    columns = core_data.columns
    column_types = { 
        col: "categorical" if core_data[col].dtype == "object" else "quantitative"
        for col in columns
    }

    dataframe = core_data
    # Generate a prompt for the OpenAI API to create the analysis code
    code_prompt = f"""
    You are a data analysis assistant. A user has asked you to: {query}
    The dataset is provided in a variable named 'dataframe'. 
    The dataframe has the following columns: {column_types}.

    Provide Python code using the 'dataframe' variable to perform the analysis, 
    and include print(...) statements to display the output directly.
    only use print(...on the output of the analysis and some strings explaining it concisely. 
    You should provide only the python code. 
    In performing the analysis, trim down the dataframe to only relevant columns / rows for this query and identifying each row as the dataset is very large.
    At the end of the code, set dataframe to the modified dataframe. 

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
        output, modified_df = execute_panda_dataframe_code(gpt_generated_code, dataframe)

        # return output as json object with description
        output_object = {
            "description": output
        }

        # Update core_data with the modified DataFrame explicitly here
        core_data = modified_df

        return output_object


    except openai.RateLimitError as e:
        logger.error(f"Rate limit error: {e}")
        raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded.")

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=500, detail="Error decoding JSON response.")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred.")

# TOOL 3: Tool to generate a table from the dataset
def tableTool(query: str, data: dict):
    # TODO later

    # Prompt the model to generate a table that will be modeled in remark gfm off of the dataset
    prompt = f"""
    You are a data assistant. The user has requested a table based on the dataset. The user has asked you to: {query}

    Please provide a markdown table that can be rendered in Remark GFM based on the dataset.

    You will provide the response in this JSON format:
    {{
        "table": "The markdown table to be rendered in Remark GFM"
        "description": "Description of the table features and analysis."
    }}
    
    This is is the data that has been passed to you to put in a table: {json.dumps(data, indent=2)}.

    """

    try:
        # Call OpenAI API to generate the table 
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
        table_response_json = json.loads(response_content)

        # Extract description and chartSpec from response
        # Extract the description and chartSpec

        table = table_response_json.get("table", "")
        
        return table
    
    except openai.RateLimitError as e:
        logger.error(f"RateLimitError: {str(e)}")
        raise HTTPException(status_code=429, detail=f"OpenAI API rate limit exceeded.")

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error decoding JSON response.")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error occurred.")

# JSON description of tableTool
tableJSON = {
    "name": "tableTool",
    "description": "Generate a table based on data and a user prompt. Use this function whenever a user requests a table based on a dataset. You should provide this function with only the data needed for the table!",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question for the function to answer, well defined and well formatted as a string. Sometimes it may need to be expanded on from the user's original query."
            },
            "data": {
                "type": "object",
                "description": "The dataset that you want to be used in your table. This should be a dictionary of the data you want to be used in the table."
            }
        },
        "required": ["query", "data"],
        "additionalProperties": False
    }
}

# JSON description of vegaLiteTool
vegaLiteJSON = {
    "name": "vegaLiteTool",
    "description": "Generate a Vega-Lite chart specification based on a dataset and a user prompt. Use this function whenever a user requests a data visualization or chart. This function only accepts a query!",
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
    "dataAnalysisTool": dataAnalysisTool,
    "tableTool": tableTool
}

# Chat functionality with JSON 
def chat(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

@app.post("/query/", response_model=QueryResponse)
async def query_openai(request: QueryRequest):
    global core_data
    core_data = pd.DataFrame(request.data)

    # Extract the columns and their types from the dataset
    columns = core_data.columns
    column_types = {
        col: "categorical" if core_data[col].dtype == "object" else "quantitative"
        for col in columns
    }

    # Define a ReAct assistant prompt 
    system_prompt = f'''
        You are a data assistant with access to a dataset containing the following columns: {column_types}.
        Utilize the tools provided: {tools}, to answer user queries.

        Workflow:
        1. Question: The input question to be answered.
        2. Thought: Detail your thought process for addressing the question.
        3. Action: Specify the tool used from [{tool_map}]. Use "no tool" if none is required.
        4. Action Input: Provide arguments in JSON format ({{"arg name": "arg value"}}). If no tool is used, use an empty JSON object {{}}.
        Ensure that inputs conform to the schema:
        - String: query
        5. Observation: Outcome of the action you executed.
        6. Final Answer: Provide a comprehensive answer to the original question, incorporating any observations and insights.

        IMPORTANT: The dataset is very long and the VegaLiteTool may struggle to produce a chart off of the full dataset, so please filter the data by querying the dataAnalysisTool first which can reduce the dataset size to only what you need!

        Responses should be structured in valid JSON format, all fields are required though some may be empty:

            "Thought": "Your thought process.",
            "Action": "The tool being used.",
            "Action Input": {{"arg name": "arg value"}},
            "Observation": "Result of the action chosen above, wait to fill in until you have the output.",
            "Final Answer": "Comprehensive answer to the query.",
            "chartSpec": {"dict"},  Fill in after receiving the output of vegaLiteTool - should be the EXACT 'chartSpec' output from the tool
            "table": ""  Markdown table to be rendered, fill in after receiving the output of the tableTool

        Note: Do not fill any of these in until you have recieved the output of your chosen action, only fill out fields that are relevant. 
        All fields must be present, even if they are empty strings or an empty dict. 

        Guidelines:
        - Always initiate with a Thought.
        - Prioritize using dataAnalysisTool first, followed by tableTool, and vegaLiteTool if visualization is required.
        - If a chart is displayed, describe it in the Final Answer.
    '''

    # Initial messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.prompt}
    ]

    description = "No final answer provided."
    chartSpec = {}
    table = ""

    max_iterations = 10
    iteration = 0

    # ReAct loop 
    while iteration < max_iterations:
        response_message = chat(messages)
        print_red("Response: ", response_message)
        messages.append({"role": "assistant", "content": response_message})

        match = re.search(r'"Action"\s*:\s*"([^"]+)",\s*"Action Input"\s*:\s*(\{[^}]+\})', response_message)
        if match:
            action_name = match.group(1)
            if action_name == "no tool":
                break  # Stop if no tool is needed

            action_input = json.loads(match.group(2))
            logger.info(f"Model selected tool: {action_name}")

            function_to_call = tool_map.get(action_name)
            if action_name == "vegaLiteTool" or action_name == "dataAnalysisTool":
                action_input["request"] = request.model_dump()

            # make work for table tool too 
            if function_to_call:
                result = function_to_call(**action_input)
                observation = f"Observation: action name: {action_name}, action_input: {json.dumps(action_input)}, result: {result['description']}"
                messages.append({"role": "assistant", "content": observation})

            # Check for chart specification 
            if "chartSpec" in result and tool_map.get(action_name) == vegaLiteTool:
                chartSpec = result['chartSpec']

            # Check for table
            if "table" in result and tool_map.get(action_name) == tableTool:
                table = result['table']
                break
        
            # Check for final answer
            if "no tool" in result['action']:
                final_answer = str(json.loads(response_message)["Final Answer"])
                description = final_answer

        iteration += 1

    # Return outside the loop after all actions and iterations are processed
    return QueryResponse(
        description=description,
        table=table,
        chartSpec=chartSpec
    )

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "API is running"}

# Port handling for Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")

