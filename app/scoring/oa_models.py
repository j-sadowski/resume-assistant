import logging
from typing import Optional
import yaml

from openai import OpenAI

from app.config import OPENAI_API_KEY
from app.datamodels.models import ComparisonExtract, WorkflowReqs, JDScore, ResumeSuggestions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# $0.10 per mil Smallest, cheapest for prototyping
# model = "gpt-4.1-nano-2025-04-14"
model = "gpt-4.1-mini-2025-04-14" #$0.40 per mil
# model = "gpt-4.1-2025-04-14" #$2.00 per million
logger.info(f"Starting OpenAI backend with model: {model}")

client = OpenAI(api_key=OPENAI_API_KEY)

# Load prompts from the YAML file
with open("app/scoring/oa_prompts.yaml", "r") as f:
    prompts = yaml.safe_load(f)


def get_prompt(prompt_name: str, message_type: str) -> str:
    """A helper function to get a specific prompt message."""
    return prompts['prompts'][prompt_name][message_type]


def formatted_chat_completion(system_prompt: str, user_prompt: str, response_format, temperature=1.0):
    completion = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {"role": "user", "content": user_prompt}
        ],
        text_format=response_format,
        temperature=temperature
    )
    result = completion.output_parsed
    return result


def basic_chat_completion(system_prompt: str, user_prompt: str, temperature=1.0) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    result = completion.choices[0].message.content
    return result


def check_request(prompt: str) -> ComparisonExtract:
    logger.info("Checking prompt validity")
    system_prompt = get_prompt("check_request", "system_message")
    result = formatted_chat_completion(system_prompt=system_prompt, user_prompt=prompt,
                                       response_format=ComparisonExtract)
    logger.info("Check complete!")
    return result


def extract_reqs(prompt: str) -> WorkflowReqs:
    logger.info("Starting prompt extraction")
    system_prompt = get_prompt("extract_reqs", "system_message")
    result = formatted_chat_completion(system_prompt=system_prompt, user_prompt=prompt,
                                       response_format=WorkflowReqs, temperature=0.0)
    logger.info("Extraction complete!")
    return result


def extract_tailoring(resume_text: str, job_description: str) -> str:
    logger.info("Starting tailoring extraction")
    system_prompt = get_prompt("extract_tailoring", "system_message")
    user_prompt = (
        f"Resume:\n---\n{resume_text}\n---\n\n"
        f"Job Description:\n---\n{job_description}\n---\n\n"
        "How tailored is this resume to the job description?"
    )
    result = basic_chat_completion(system_prompt=system_prompt, user_prompt=user_prompt,
                                   temperature=0.0)
    logger.info("Extraction complete!")
    return result


def score_resume(resume_text: str, job_description: str) -> JDScore:
    """
    Evaluates the suitability of a resume for a specific job description.

    Sends both the resume and the job description to LLM, which returns a numerical
    score (0-10) indicating how well the resume matches the job requirements, along with a brief explanation.
    A score of 10 means a perfect fit; 0 means no fit. The evaluation considers skills, experience,
    qualifications, and alignment with the role's responsibilities.

    Args:
        resume_text (str): The plain text content of the candidate's resume.
        job_description (str): The plain text content of the job description.

    Returns:
        JDScore: An object containing the suitability score and an explanation.
    """
    system_prompt = get_prompt("score_resume", "system_message")
    user_prompt = (
        f"Resume:\n---\n{resume_text}\n---\n\n"
        f"Job Description:\n---\n{job_description}\n---\n\n"
        "Score this resume against the job description (0-10):"
    )

    try:
        result = formatted_chat_completion(system_prompt=system_prompt, user_prompt=user_prompt,
                                           response_format=JDScore, temperature=0.0)
        logger.info("Resume scoring successful!")
    except Exception as e:
        logger.error(f"Failed to score resume: {e}")
        result = JDScore(score=-1, explanation="Comparison failed")
    return result


def summarize_gaps(explanation: str) -> str:
    """
    Analyzes an explanation to extract missing skills or experiences.

    This function sends a list of explanations to an LLM, which returns
    a concise bullet-point list of specific skills or experiences that are identified as missing
    or could be improved upon for a higher suitability score. The output is intended to help
    candidates understand what areas of their resume could be strengthened to better match job requirements.

    Args:
        explanations (List[str]): List of rationale strings, each describing aspects of a candidate's profile
                                  in relation to a job description.

    Returns:
        str: A bullet-point list of missing skills or experiences, as identified by the LLM.
    """
    logger.info("Starting gap summarizer")

    explanation = f"Rationale: {explanation}"
    system_prompt = get_prompt("summarize_gaps", "system_message")
    user_prompt = (
        f"Analyze the following rationale to identify missing skills or experiences:\n"
        f"{'\n--\n'.join(explanation)}\n--\n\n"
        "List the identified gaps:"
    )

    try:
        result = basic_chat_completion(system_prompt=system_prompt, user_prompt=user_prompt,
                                       temperature=0.0)
        logger.info("Gap summarizaton complete")
    except Exception as e:
        logger.error(f"Failed to identify gaps resume: {e}")
        result = f"Unable to analyze gaps: {e}"
    return result


def suggest_edits(resume_text: str, job_description: str, gaps: Optional[str]) -> ResumeSuggestions:
    system_prompt = get_prompt("suggest_edits", "system_message")
    user_prompt = (
        "Provide edit suggestions for my resume:"
        f"Resume:\n---\n{resume_text}\n---\n\n"
        f"Job Description:\n---\n{job_description}\n---\n\n"
    )
    if gaps:
        user_prompt += f"A separate analysis indicated these gaps: \n---\n{gaps}\n---\n\n"

    try:
        result = formatted_chat_completion(system_prompt=system_prompt, user_prompt=user_prompt,
                                           response_format=ResumeSuggestions, temperature=0.0)
        logger.info("Resume edit suggestions request successful!")
    except Exception as e:
        logger.error(f"Failed to score resume: {e}")
        result = ResumeSuggestions(suggestions="Comparison failed")
    return result
