import logging
from pathlib import Path
from typing import Optional

from app.datamodels.models import ResumeSuggestions, EvaluateSuggestions
from app.utils.oa_utils import formatted_chat_completion
from app.utils.utils import load_yaml_prompts, get_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

workflow_prompts = load_yaml_prompts(Path("app/workflow/oa_prompts.yaml"))
agent_prompts = load_yaml_prompts(Path("app/agent/oa_prompts.yaml"))


def suggest_edits(resume_text: str, job_description: str, gaps: Optional[str], context="") -> ResumeSuggestions:
    """
    Suggest edits to improve a resume based on a job description and identified gaps using the LLM.

    Args:
        resume_text (str): The plain text of the candidate's resume.
        job_description (str): The plain text of the job description.
        gaps (Optional[str]): Identified gaps or missing skills/experiences.

    Returns:
        ResumeSuggestions or None: Suggestions for resume improvements, or None on failure.
    """
    system_prompt = get_prompt(
        workflow_prompts, "suggest_edits", "system_message")
    user_prompt = (
        "Provide edit suggestions for my resume:"
        f"Resume:\n---\n{resume_text}\n---\n\n"
        f"Job Description:\n---\n{job_description}\n---\n\n"
    )
    if gaps:
        user_prompt += f"A separate analysis indicated these gaps: \n---\n{gaps}\n---\n\n"
    if context:
        user_prompt += f"\nContext:{context}"

    try:
        result = formatted_chat_completion(system_prompt=system_prompt, user_prompt=user_prompt,
                                           response_format=ResumeSuggestions, temperature=0.0)
        logger.info("Resume edit suggestions request successful!")
    except Exception as e:
        logger.error(f"Failed to create resume suggestions: {e}")
        result = ResumeSuggestions(suggestions="Comparison failed")
    return result


def evaluate_edits(resume_text: str, job_description: str, suggestions: ResumeSuggestions) -> EvaluateSuggestions:

    system_prompt = get_prompt(
        agent_prompts, "evaluate_edits", "system_message")
    user_prompt = (
        f"Resume:\n---\n{resume_text}\n---\n\n"
        f"Job Description:\n---\n{job_description}\n---\n\n"
        f"Resume Suggestions:\n---\n{suggestions.suggestions}\n---\n\n"
    )
    try:
        result = formatted_chat_completion(system_prompt=system_prompt, user_prompt=user_prompt,
                                           response_format=EvaluateSuggestions)
        logger.info("Resume Suggestion eval request successful")
    except Exception as e:
        logger.error(f"Failed to evaluate resume suggestions: {e}")
        result = EvaluateSuggestions(
            evaluation=False, feedback=f"Failed to evaluate resume suggestions: {e}")
    return result


def loop_suggestions_eval(resume_text: str,
                          job_description: str,
                          gaps: Optional[str],
                          max_iter=5) -> ResumeSuggestions:
    logger.info("Starting Edit Suggestion/Refinement agentic loop")
    memory = []
    result = suggest_edits(resume_text=resume_text,
                           job_description=job_description,
                           gaps=gaps)
    memory.append(result)
    i = 0
    while i <= max_iter:
        evaluation = evaluate_edits(resume_text=resume_text,
                                    job_description=job_description,
                                    suggestions=result)

        if evaluation.evaluation:
            logger.info("Edit Suggestion/Refinment agentic loop complete")
            return result
        context = "\n".join([
            "Previous attempts:",
            *[f"- {m}" for m in memory],
            f"\nFeedback: {evaluation.feedback}"
        ])

        result = suggest_edits(
            resume_text=resume_text, job_description=job_description, gaps=gaps, context=context)
        memory.append(result)
        i += 1
    logger.info(f"Edit Suggestion/Refinement loop exited after {max_iter} results")
    return result
