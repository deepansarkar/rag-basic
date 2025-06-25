import os
import requests
import time
from dotenv import load_dotenv

# Load environment variables from a .env file into the system environment
load_dotenv()

# Retrieve required configuration values from the environment
API_KEY = os.getenv("OPENROUTER_API_KEY")       # API key for OpenRouter authentication
API_URL = os.getenv("OPENROUTER_API_URL")       # URL endpoint for the OpenRouter API
MIN_TRIES = int(os.getenv("MIN_TRIES"))         # Number of retry attempts
LLM_MODEL = os.getenv("LLM_MODEL")              # Language model to use

# Define the headers required for the API request
headers = {
    "Authorization": f"Bearer {API_KEY}",        # Bearer token for API authentication
    "Content-Type": "application/json",          # Specify content type as JSON
}

def query_openrouter(question, context):
    """
    Calls the OpenRouter API with a question and context to retrieve an answer.
    If the API call fails, it retries multiple times (up to MIN_TRIES).
    
    Parameters:
        question (str): The question to ask the model.
        context (str): The context in which the question should be answered.
        
    Returns:
        str: The model's response or an error message if all attempts fail.
    """

    # Create the prompt that guides the model using a Retrieval-Augmented Generation (RAG) style
    prompt = f"""Use the context below to answer the question. If the answer isn't in the context, say "I am not able to answer based on provided context".

    Context:
    {context}

    Question: {question}
    """

    # Construct the data payload to send to the OpenRouter API
    data = {
        "model": LLM_MODEL,  # Specifies the model to use
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant who answers question based on context from provided data. Reply with concise and accurate information in a single paragraph."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    # Retry loop: attempt the request up to MIN_TRIES times
    for attempt in range(1, MIN_TRIES + 1):
        try:
            # Make the POST request to the OpenRouter API
            response = requests.post(API_URL, headers=headers, json=data, timeout=15)

            # If the request was successful (HTTP 200 OK), parse and return the model's response
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                # Log the HTTP status code for non-successful attempts
                print(f"Attempt {attempt}: Status {response.status_code}")
        except Exception as e:
            # Log any exceptions that occur during the request
            print(f"Attempt {attempt}: Exception - {e}")

        # Wait 1 second before retrying to avoid rapid repeated requests
        time.sleep(1)

    # If all attempts fail, return an error message
    return "Failed to get a valid response after multiple attempts."
