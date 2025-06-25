import os
import requests
from dotenv import load_dotenv

# Load environment variables from a .env file into the system environment
load_dotenv()

# Retrieve required configuration values from the environment
API_KEY = os.getenv("OPENROUTER_API_KEY")       # API key for OpenRouter authentication
API_URL = os.getenv("OPENROUTER_API_URL")       # URL endpoint for the OpenRouter API\
LLM_MODEL = os.getenv("LLM_MODEL")              # Language model to use

# Define a function to query the OpenRouter API using a single prompt
def ask_openrouter(question: str) -> str:
    # Define the request headers, including authorization and content type
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # Construct the request body including system prompt and user query
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Reply with concise and accurate information in a single paragraph."},
            {"role": "user", "content": question}
        ]
    }

    # Attempt to send a single POST request to the OpenRouter API
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=15)

        # If the response is successful (HTTP 200), return the content
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()

        # If the response is an error, print the status and message
        else:
            print(f"Error {response.status_code} — {response.text}")
            return "Failed to get a valid response from OpenRouter."

    # Handle network or request exceptions gracefully
    except Exception as e:
        print(f"Exception — {e}")
        return "An error occurred while querying OpenRouter."

# Entry point to run the function if the script is executed directly
if __name__ == "__main__":
    # Prompt the user for a question via the command line
    user_question = input("Ask your question: ")

    # Call the ask_openrouter function and display the result
    print(ask_openrouter(user_question))