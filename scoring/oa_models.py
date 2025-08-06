import logging
from typing import List

from openai import OpenAI

from config import OPENAI_API_KEY
from datamodels.models import ComparisonExtract, WorkflowReqs, JDScore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# model = "gpt-4.1-nano-2025-04-14" #$0.10 per mil Smallest, cheapest for prototyping
model = "gpt-4.1-mini-2025-04-14" #$0.40 per mil
# model = "gpt-4.1-2025-04-14" #$2.00 per million

client = OpenAI(api_key=OPENAI_API_KEY)

def check_request(prompt: str) -> ComparisonExtract:
    return ComparisonExtract(is_valid=True, confidence=0.9, rationale="PassThrough Value")

def extract_reqs(prompt: str) -> WorkflowReqs:
    logger.info("Starting prompt extraction")
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Extract keywords to search, the city to search in, and optional hybrid / limit parameters",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=WorkflowReqs,
        temperature=0.0
    )
    result = completion.choices[0].message.parsed
    logger.info("Extraction complete!")
    print(result)
    return result



def resume_summarizer(resume: str) -> ResumeDigest:
    """
    Summarize a resume to extract key keep the important bits for the for loop analysis

    This function sends the provided resume text to an LLM, which returns a concise summary
    highlighting the most important keywords and phrases. The summary is intended to capture
    the core qualifications, skills, and experience from the resume for downstream analysis.
    
    Leaving this in for reference, but it makes the results worse. Probably needs to be replaced with
    a tokenization step or something.

    Args:
        resume (str): The plain text content of the candidate's resume.

    Returns:
        ResumeDigest: An object containing the summarized resume.
    """
    logger.info("Starting resume summarizer")
    

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Summarize the provided resume, extract the important key words and phrases.",
            },
            {"role": "user", "content": resume},
        ],
        response_format=ResumeDigest,
    )
    result = completion.choices[0].message.parsed
    logger.info("Summary complete!")
    print(result.summary)
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

    system_prompt = (
        "You are an expert resume evaluator. Your task is to score a resume's suitability "
        "for a given job description on a scale of 0 to 10. "
        "A score of 10 indicates a perfect fit, and 0 indicates no fit. "
        "Consider all aspects: skills, experience, qualifications, and alignment with the role's responsibilities. "
        "Provide the numerical score as an float and a short explanation."
    )

    user_prompt = (
        f"Resume:\n---\n{resume_text}\n---\n\n"
        f"Job Description:\n---\n{job_description}\n---\n\n"
        "Score this resume against the job description (0-10):"
    )

    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format=JDScore
        )
        logger.info("Resume scoring successful!")
        result = response.choices[0].message.parsed
    except Exception as e:
        logger.error(f"Failed to score resume: {e}")
        result = JDScore(score=-1, explanation="Comparison failed")
    return result

