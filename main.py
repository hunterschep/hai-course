from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI(debug=True)

client = OpenAI()

# Load OpenAI API key from environment variable
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    raise HTTPException(status_code=500, detail="OpenAI API key is missing")

openai.api_key = api_key

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define request and response models
class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    response: str

@app.post("/query", response_model=QueryResponse)
async def query_openai(request: QueryRequest):
    try:
        # Instead of querying OpenAI, just return the dummy response
        return QueryResponse(response="I am a simple bot and do not have any responses yet!")
    
    except openai.RateLimitError as e:  # Catch API errors
        raise HTTPException(status_code=499, detail=f"OpenAI API error: {str(e)}")
    
    except Exception as e:  # Handle other potential errors
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "API is running"}
