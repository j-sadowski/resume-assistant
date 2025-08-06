import logging
import os

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


load_dotenv()

AI_BACKEND = os.getenv("AI_BACKEND", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if AI_BACKEND == "openai" and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set, but OpenAI backend selected.")

logger.info(f"Using AI backend: {AI_BACKEND}")