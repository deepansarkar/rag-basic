import os
import requests
import time
from dotenv import load_dotenv

# Load environment variables from a .env file into system environment
load_dotenv()

# Retrieve required configuration values from the environment
API_KEY = os.getenv("OPENROUTER_API_KEY")   # API key for OpenRouter
API_URL = os.getenv("OPENROUTER_API_URL")   # API endpoint URL
MIN_TRIES = int(os.getenv("MIN_TRIES"))     # Number of retry attempts
LLM_MODEL = os.getenv("LLM_MODEL")          # Language model name

# Prepare the headers for the API request, including authentication
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

def query_openrouter(question, context, logger):
    """
    Calls the OpenRouter API with a question and context to retrieve an answer.
    Includes robust retry logic with logging.

    Parameters:
        question (str): The question to ask the model.
        context (str): Relevant context for the question.
        logger: Logger object for structured logging.

    Returns:
        str: The language model's answer, or an error message after all attempts fail.
    """

    # Construct the prompt following RAG style: embed the context and question together
    prompt = f"""Use the context below to answer the question. If the answer isn't in the context, say "I am not able to answer based on provided context".

Context:
{context}

Question: {question}
"""

    # Create the JSON payload for the API request
    data = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant who answers questions based on provided context. Respond concisely in one paragraph."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    # Log the start of the API interaction
    logger.info(f"Sending query to OpenRouter model '{LLM_MODEL}' with question: {question}")

    # Try to make the API request up to MIN_TRIES times
    for attempt in range(1, MIN_TRIES + 1):
        try:
            # Attempt to POST the request
            response = requests.post(API_URL, headers=headers, json=data, timeout=15)

            # Check if the HTTP response was successful
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"].strip()

                logger.info(f"Received successful response on attempt {attempt}.")

                return answer
            else:
                # Non-200 HTTP response
                logger.warning(f"Attempt {attempt}: Received status code {response.status_code}. Response: {response.text}")

        except requests.exceptions.Timeout:
            # Specific handling for timeouts
            logger.error(f"Attempt {attempt}: Request timed out.")
        except Exception as e:
            # Catch all other exceptions
            logger.error(f"Attempt {attempt}: Exception occurred - {e}", exc_info=True)

        # Wait a short time before retrying
        time.sleep(1)

    # If all retries fail, log the final outcome
    logger.error(f"Failed to get valid response after {MIN_TRIES} attempts.")

    return "Failed to get a valid response after multiple attempts."
