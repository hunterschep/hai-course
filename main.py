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
def vegaLiteTool(request: dict):
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
        
        The user has requested visualization with the question: {request["prompt"]}.
        
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
        logger.info(f"Generated prompt: {prompt}")

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
def dataAnalysisTool(request: dict) -> str:
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
    query = request.get("prompt")
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

# JSON description of vegaLiteTool
vegaLiteJSON = {
    "name": "vegaLiteTool",
    "description": "Generate a Vega-Lite chart specification based on a dataset and a user prompt. Use this function whenever a user requests a data visualization or chart.",
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "object",
                "description": "Object containing user query and dataset information.",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "User's request or question describing the desired chart."
                    },
                    "data": {
                        "type": "object",
                        "description": "Object representing the dataset, where keys are column names and values are arrays of data values.",
                        "additionalProperties": True
                    }
                },
                "required": ["prompt", "data"]
            }
        },
        "required": ["request"],
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
            "request": {
                "type": "object",
                "description": "Object containing user query and dataset information.",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "User's request or question describing the desired chart."
                    },
                    "data": {
                        "type": "object",
                        "description": "Object representing the dataset, where keys are column names and values are arrays of data values.",
                        "additionalProperties": True
                    }
                },
                "required": ["prompt", "data"]
            }
        },
        "required": ["request"],
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

@app.post("/query/", response_model=QueryResponse)
async def query_openai(request: QueryRequest):
    logger.info(f"Received a new request with prompt: {request.prompt}")

    # Conversation history for the API
    messages = [
        {"role": "system", "content": "You are a data analysis and visualization assistant."},
        {"role": "user", "content": request.prompt}
    ]

    try:
        # Initial model call to assess if a tool is needed
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            functions=tools  # Pass in JSON schemas for the tools
        )

        # Extract response and check for function call
        response_message = response.choices[0].message
 
        if response_message.function_call:
            # Tool call specifics
            tool_name = response_message.function_call.name
            arguments = json.loads(response_message.function_call.arguments)

            # Ensure 'data' is included in the arguments if available in the request
            if hasattr(request, "data") and request.data:
                arguments["request"]["data"] = request.data

            logger.info(f"Model selected tool: {tool_name} with arguments: {arguments}")

            # Locate and execute the selected tool
            function_to_call = tool_map.get(tool_name)
            if function_to_call:
                output = function_to_call(**arguments)

                # Add tool's response to the message history
                messages.append({
                    "role": "function",
                    "name": tool_name,
                    "content": json.dumps(output)
                })
                
                # If the tool is dataAnalysis tool, we need to return it back to the model to generate the analysis 
                analysis_prompt = f"Provide the result of the code generated above in a json object called 'description'."
                if tool_name == "dataAnalysisTool":
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages
                    )
                    response_message = response.choices[0].message

                # Need to make sure this accurately returns the output of the data analysis tool! 
                # Respond with tool output based on tool type
                print(output)
                return QueryResponse(
                    description=output.get("description") if tool_name == "vegaLiteTool" else response_message.content,
                    chartSpec=output.get("chartSpec", {}) if tool_name == "vegaLiteTool" else {}
                )

        # If no tool was used, return direct model response
        messages.append(response_message)
        return QueryResponse(
            description=response_message.content,
            chartSpec={}
        )

    except openai.RateLimitError as e:
        logger.error(f"Rate limit error: {e}")
        raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded.")

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=500, detail="Error decoding JSON response.")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "API is running"}

# Ensure proper port handling for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
