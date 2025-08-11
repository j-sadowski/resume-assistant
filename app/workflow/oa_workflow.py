import logging
from pathlib import Path
from typing import Optional, Union

from app.datamodels.models import ComparisonExtract, WorkflowReqs, JDScore, ResumeSuggestions
from app.utils.oa_utils import formatted_chat_completion, basic_chat_completion
from app.utils.utils import load_yaml_prompts, get_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

prompts = load_yaml_prompts(Path("app/scoring/oa_prompts.yaml"))

def check_request(prompt: str) -> Union[ComparisonExtract, None]:
    """
    Validate a prompt using the LLM and return a ComparisonExtract object.

    Args:
        prompt (str): The prompt to validate.

    Returns:
        ComparisonExtract or None: The validation result, or None on failure.
    """
    logger.info("Checking prompt validity")
    system_prompt = get_prompt(prompts, "check_request", "system_message")
    try:
        result = formatted_chat_completion(system_prompt=system_prompt, user_prompt=prompt,
                                           response_format=ComparisonExtract)
        logger.info("Check complete!")
    except Exception as e:
        logger.error(f"Failed to check prompt validity: {e}")
        result = ComparisonExtract(
            is_valid=False, confidence=0.0, rationale=f"Validity check failed: {e}")
    return result


def extract_reqs(prompt: str) -> Union[WorkflowReqs, None]:
    """
    Extract workflow requirements from a prompt using the LLM.

    Args:
        prompt (str): The prompt to analyze.

    Returns:
        WorkflowReqs or None: The extracted requirements, or None on failure.
    """
    logger.info("Starting prompt extraction")
    system_prompt = get_prompt(prompts, "extract_reqs", "system_message")
    try:
        result = formatted_chat_completion(system_prompt=system_prompt, user_prompt=prompt,
                                           response_format=WorkflowReqs, temperature=0.0)
        logger.info("Extraction complete!")
    except Exception as e:
        logger.error(f"Failed to extract variables from prompt: {e}")
        result = WorkflowReqs(score_resume=False, score_confidence=0.0, predict_success=False,
                              predict_confidence=0.0, suggest_edits=False, edit_confidence=0.0,
                              rationale=f"Failed to extract variables: {e}")
    return result


def extract_tailoring(resume_text: str, job_description: str) -> Union[str, None]:
    """
    Assess how well a resume is tailored to a specific job description using the LLM.

    Args:
        resume_text (str): The plain text of the candidate's resume.
        job_description (str): The plain text of the job description.

    Returns:
        str or None: The LLM's assessment of tailoring, or None on failure.
    """
    logger.info("Starting tailoring level extraction")
    system_prompt = get_prompt(prompts, "extract_tailoring", "system_message")
    user_prompt = (
        f"Resume:\n---\n{resume_text}\n---\n\n"
        f"Job Description:\n---\n{job_description}\n---\n\n"
        "How tailored is this resume to the job description?"
    )
    try:
        result = basic_chat_completion(system_prompt=system_prompt, user_prompt=user_prompt,
                                    temperature=0.0)
        logger.info("Extraction complete!")
    except Exception as e:
            logger.error(f"Failed to extract tailoring level: {e}")
            result = ""
    return result


def score_resume(resume_text: str, job_description: str) -> Union[JDScore, None]:
    """
    Evaluates the suitability of a resume for a specific job description using the LLM.

    Args:
        resume_text (str): The plain text content of the candidate's resume.
        job_description (str): The plain text content of the job description.

    Returns:
        JDScore or None: An object containing the suitability score and an explanation, or None on failure.
    """
    system_prompt = get_prompt(prompts, "score_resume", "system_message")
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


def summarize_gaps(explanation: str) -> Union[str, None]:
    """
    Analyze an explanation to extract missing skills or experiences using the LLM.

    Args:
        explanation (str): Rationale describing aspects of a candidate's profile in relation to a job description.

    Returns:
        str or None: A bullet-point list of missing skills or experiences, or None on failure.
    """
    logger.info("Starting gap summarizer")

    explanation = f"Rationale: {explanation}"
    system_prompt = get_prompt(prompts, "summarize_gaps", "system_message")
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


def suggest_edits(resume_text: str, job_description: str, gaps: Optional[str]) -> Union[ResumeSuggestions, None]:
    """
    Suggest edits to improve a resume based on a job description and identified gaps using the LLM.

    Args:
        resume_text (str): The plain text of the candidate's resume.
        job_description (str): The plain text of the job description.
        gaps (Optional[str]): Identified gaps or missing skills/experiences.

    Returns:
        ResumeSuggestions or None: Suggestions for resume improvements, or None on failure.
    """
    system_prompt = get_prompt(prompts, "suggest_edits", "system_message")
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
