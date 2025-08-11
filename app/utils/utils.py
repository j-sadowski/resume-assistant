from pathlib import Path
from typing import Dict
import yaml

def load_yaml_prompts(path_to_prompt: Path) -> Dict[str, Dict[str, str]]:
    with open(path_to_prompt, "r") as f:
        prompts = yaml.safe_load(f)
    return prompts

def get_prompt(prompts: Dict[str, Dict[str, Dict[str, str]]], prompt_name: str, message_type: str) -> str:
    """
    Retrieve a specific prompt message from the loaded YAML prompts.

    Args:
        prompt_name (str): The name of the prompt to retrieve.
        message_type (str): The type of message (e.g., 'system_message', 'user_message').

    Returns:
        str: The requested prompt message.
    """
    return prompts['prompts'][prompt_name][message_type]

