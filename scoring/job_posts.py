import logging
from typing import List

from config import AI_BACKEND
if AI_BACKEND == "openai":
    from .oa_models import score_resume, summarize_gaps
elif AI_BACKEND == "ollama":
    from .ollama_models import score_resume, summarize_gaps
else:
    raise ValueError(f"Unknown AI_BACKEND: {AI_BACKEND}. Must be 'ollama' or 'openai'.")

from datamodels.models import JobInfo


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def score_job_posts(resume: str, job_postings: List[JobInfo]) -> List[JobInfo]:
    """
    Scores a list of job postings against a candidate's resume using an LLM.

    For each job posting, this function evaluates how well the provided resume matches
    the job description by calling the score_resume function. It updates each JobInfo object
    with a suitability score and an explanation. If an error occurs during scoring, the job's
    score is set to -1.

    Args:
        resume (str): The plain text content of the candidate's resume.
        job_postings (List[JobInfo]): A list of JobInfo objects representing job postings to score.

    Returns:
        List[JobInfo]: The input list of JobInfo objects, each updated with a score and explanation.
    """
    logger.info(f"Starting resume scorer. Submitting {len(job_postings)} jobs")
    scores = []
    for i, job in enumerate(job_postings):
        logger.info(f"Submitting job {i + 1} of {len(job_postings)}")
        job_score = score_resume(resume, job.description)
        job.score = job_score.score
        job.explanation = job_score.explanation
        scores.append(job)
    return scores

def identify_resume_gaps(scores: List[JobInfo], score_threshold=7) -> str:
    explanations = [x.explanation for x in scores if x.score > score_threshold]
    if len(explanations) == 0:
        logger.info(f"No jobs scored above {score_threshold=}")
        return f"Unable to conduct gap analysis, no jobs scored above {score_threshold}"
    logger.info(f"Submitting {len(explanations)} jobs above {score_threshold} to gap analysis")
    return summarize_gaps(explanations)