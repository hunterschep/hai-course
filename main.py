from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import uvicorn
import logging

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

@app.post("/query/", response_model=QueryResponse)
async def query_openai(request: QueryRequest):
    logger.info(f"Received a new request with prompt: {request.prompt}")
    # Extract relevant information from the dataset
    if not request.data:
        return QueryResponse(response="No dataset uploaded. Please upload a dataset to generate charts.", chartSpec={})

    try:
        # Gather information from the dataset for GPT-4
        columns = list(request.data[0].keys())  # Column names
        column_types = {col: "categorical" if isinstance(request.data[0][col], str) else "quantitative" for col in columns}
        full_data = request.data # dataset

        # Create the prompt for GPT to generate a Vega-Lite chart specification
        prompt = f"""
        You are a data visualization assistant. A user has provided a dataset with the following columns:
        {json.dumps(column_types, indent=2)}. 
        Here is the full dataset: {json.dumps(full_data, indent=2)}.
        The user has asked the following question: {request.prompt}.
        
        You must generate a valid Vega-Lite JSON chart specification and a short description of the chart features like the chart type, analysis of what it displays, etc. Ensure your description is placed in a 'description' key in the JSON object.
        """

        # Log the constructed prompt
        logger.info(f"Constructed prompt: {prompt}")

        # Call OpenAI to generate the Vega-Lite specification
        gpt_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "assistant", "content": "You are a data visualization assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.25,
            n=1,
            response_format={"type": "json_object"}
        )

        # Extract the response from GPT-4
        gpt_message = gpt_response.choices[0].message.content

        # Parse the GPT-4 response as JSON
        chart_response_json = json.loads(gpt_message)

        # Extract the description and chartSpec
        description = chart_response_json.get("description", "")
        chart_spec = {key: chart_response_json[key] for key in chart_response_json if key != "description"}

        # Return the response with description and chartSpec
        return QueryResponse(
            description=description,
            chartSpec=chart_spec
        )

    except openai.RateLimitError as e:
        logger.error(f"RateLimitError: {str(e)}")
        raise HTTPException(status_code=499, detail=f"OpenAI API error: {str(e)}")

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {str(e)}")
        raise HTTPException(status_code=500, detail=f"JSON decode error: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "API is running"}

# Ensure proper port handling for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
