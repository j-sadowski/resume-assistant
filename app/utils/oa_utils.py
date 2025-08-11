import logging
from typing import Union

from openai import OpenAI

from app.config import OPENAI_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# $0.10 per mil Smallest, cheapest for prototyping
# model = "gpt-4.1-nano-2025-04-14"
model = "gpt-4.1-mini-2025-04-14"  # $0.40 per mil
# model = "gpt-4.1-2025-04-14" #$2.00 per million
logger.info(f"Starting OpenAI backend with model: {model}")

client = OpenAI(api_key=OPENAI_API_KEY)


def formatted_chat_completion(system_prompt: str, user_prompt: str, response_format, temperature=1.0):
    """
    Send a formatted chat completion request to the LLM and parse the response.

    Args:
        system_prompt (str): The system prompt to provide context to the LLM.
        user_prompt (str): The user's input prompt.
        response_format: The expected response format or class for parsing.
        temperature (float, optional): Sampling temperature for the LLM. Defaults to 1.0.

    Returns:
        Parsed response in the specified format.
    """
    completion = client.responses.parse(
        model=model,
        input=[
            {
                "role": "developer",
                "content": system_prompt
            },
            {   
                "role": "user", 
                "content": user_prompt
            }
        ],
        text_format=response_format,
        temperature=temperature
    )
    result = completion.output_parsed
    if not result:
        raise ValueError("API call returned a None result.")
    return result


def basic_chat_completion(system_prompt: str, user_prompt: str, temperature=1.0) -> str:
    """
    Send a basic chat completion request to the LLM and return the response as a string.

    Args:
        system_prompt (str): The system prompt to provide context to the LLM.
        user_prompt (str): The user's input prompt.
        temperature (float, optional): Sampling temperature for the LLM. Defaults to 1.0.

    Returns:
        str or None: The LLM's response as a string, or None if unavailable.
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "developer", 
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=temperature
    )
    result = completion.choices[0].message.content
    if not result:
        raise ValueError("API call returned a None result.")
    return result

