# Resume Assistant

Resume Assistant is a command-line tool that uses LLMs to assess your resume against a job posting. It identifies strengths and weaknesses, predicts your chance of getting an interview, and suggests improvements.

If you are interested in using the Ollama backend, make sure you have Ollama installed and the proper model specified.

## Application Workflow

![Diagram of how the application calls various components](./workflow.png)

## Features
- **Resume Scoring:** Get a suitability score for your resume against a job description.
- **Gap Analysis:** Find missing skills or experiences to improve your fit.
- **Tailoring Assessment:** See how well your resume is tailored to the job.
- **Success Prediction:** Estimate your probability of getting an interview (based on score and degree of tailoring).
- **Edit Suggestions:** Receive actionable suggestions to strengthen your resume.

## Requirements
- Only tested on Python 3.13.5
- OpenAI API key
- Apify API key

## Setup
1. **Clone the repository:**
	```sh
	git clone https://github.com/j-sadowski/resume-assistant.git
	cd resume-assistant
	```
2. Set your up `.env` file:
    ```
    AI_BACKEND=either "openai" or "ollama"
    OPENAI_API_KEY=your_openai_api_key_here
    ```

## Usage
Prepare your resume and job posting as plain text files. Then run:

```sh
uv run main.py -r path/to/resume.txt -j path/to/job_posting.txt -p "Assess my resume, predict whether I'll get an interview, and suggest improvements."
```

### Arguments
- `-r`, `--resume_path`: Path to your resume (.txt)
- `-j`, `--job_posting`: Path to the job posting (.txt)
- `-p`, `--prompt`: The LLM prompt (e.g., "Score my resume and suggest improvements.")

## Example Output
```
8.5
Your resume matches most of the required skills, but is missing experience with cloud technologies.

Areas of improvement
- Add more details about cloud experience
- Highlight leadership roles
```