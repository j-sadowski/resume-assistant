import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from pprint import pprint

from datamodels.models import JobInfo
from scoring.prompt_extraction import check_and_extract
from scoring.job_posts import score_resume, summarize_gaps, suggest_edits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path.cwd() / "data"
CACHE_DIR.mkdir(exist_ok=True)

def cache_data(score: JobInfo, gap_summary: str) -> None:
    """
    Saves all data to a cache file.
    Args:
        scores (List[JobInfo]): List of JobInfo objects containing job details and scores.
        gap_summary (str): Summary of areas where the resume could be improved.
    Side Effects:
        - Saves all job scores and details to a timestamped JSON file in the cache directory.

    """
    dt_string = datetime.now(timezone.utc).strftime(format="%Y%m%d-%H%M%S")
    outfile = CACHE_DIR / f"jobs_{dt_string}.json"
    logger.info(f"Saving data to {outfile}")
    jobs_d = {
        "query_date": dt_string,
        "job": score.model_dump(),
        "areas_of_improvement": gap_summary
    }
    with open(outfile, "w") as f:
        json.dump(jobs_d, f, indent=4)


def display_output(score: JobInfo, gap_summary: str) -> None:
    """
    Displays the top job matches and areas for improvement

    Args:
        scores (List[JobInfo]): List of JobInfo objects containing job details and scores.
        gap_summary (str): Summary of areas where the resume could be improved.
        top_n (int, optional): Number of top jobs to display. Defaults to 5.

    Side Effects:
        - Prints the top N job matches and their explanations to the console.
        - Prints the gap summary to the console.
    """
    print("")
    print(score.score)
    print(score.explanation)
    print("")
    print("Areas of improvement")
    print(gap_summary)


def run_workflow(resume: str, job_posting: str, prompt: str) -> None:
    """
    Executes the main workflow for job searching and evaluation.

    Steps:
        1. Take a job posting and a resume, evaluates against it.
        2. Provides suggestions on where to strengthen said resume

    Args:
        resume (str): The contents of the user's resume in plain text.
        job_posting (str): The text of a job posting

    Returns:
        None
    """\
    
    request_values = check_and_extract(prompt)
    gap_summary = ""
    if request_values.score_resume:
        score = score_resume(resume, job_posting)
        gap_summary = summarize_gaps(score)
        display_output(score, gap_summary)
        cache_data(score, gap_summary)
    if request_values.predict_success:
        if not request_values.score_resume:
            score = score_resume(resume, job_posting)
        success_probability = predict_success(score, job_posting)
        display_success(success_probability)
    if request_values.suggest_edits:
        edits = suggest_edits(resume, job_posting, gap_summary)
        pprint(edits.suggestions)
    logger.info("Script complete")

def extract_txt_file(fpath: Path) -> str:
    content = ""
    try:
        with open(fpath) as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Unable to extract content from {fpath}! {e}")
    return content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM based job searches given a prompt")
    parser.add_argument("-r", "--resume_path", type=Path, help="Path to local resume, currently only .txt format", required=True)
    parser.add_argument("-j", "--job_posting", type=Path, help="Path to local job posting, currently only .txt format", required=True)
    parser.add_argument("-p", "--prompt", type=str, help="The LLM prompt", required=True)
    args = parser.parse_args()

    logger.info(f"Reading resume from {args.resume_path}")
    resume = extract_txt_file(args.resume_path)
    if not resume:
        logger.info("Exiting")
        exit(1)
    logger.info(f"Reading job posting from {args.job_posting}")
    job_posting = extract_txt_file(args.job_posting)
    if not job_posting:
        logging.info("Exiting")
        exit(1)
    run_workflow(resume, job_posting, args.prompt)
