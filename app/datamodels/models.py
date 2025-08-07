from typing import Optional
from pydantic import BaseModel, Field


class ComparisonExtract(BaseModel):
    is_valid: bool = Field(description="Whether this text is describing terms for requesting assistance in writing a resume for a job description")
    confidence: float = Field(description="Confidence score that this is a request")
    rationale: str = Field(description="A concise explanation on why the prompt was given this score")

class WorkflowReqs(BaseModel):
    score_resume: bool = Field(description="Whether this text is describing a request for comparing a resume against a job description")
    score_confidence: float = Field(description="Confidence score that this contains a resume comparison request")
    predict_success: bool = Field(description="Whether this text is describing a request for predicting the success of a resume for a job description")
    predict_confidence: float = Field(description="Confidence score that this contains a predict success request")
    suggest_edits: bool = Field(description="Whether this text is describing a request for suggesting edits to the resume")
    edit_confidence: float = Field(description="Confidence score that this is contains a resume edit requess")
    rationale: str = Field(description="A concise explanation on why the prompt was given these scores")

class JDScore(BaseModel):
    """Score the JD against the resume"""
    # description: str = Field(description="The job description")
    # resume: str = Field(description="The resume")
    score: float = Field(description="Resume suitability score")
    explanation: str = Field(description="Explanation of suitability score")

class ResumeSuggestions(BaseModel):
    suggestions: str = Field(description="Suggestions on how to update the resume")

class ResumeDigest(BaseModel):
    """Summarize the resume"""
    summary: str = Field(description="Summary of the input resume")

class JobInfo(BaseModel):
    description: str = Field(description="The job description")
    resume: str = Field(description="The resume")
    score: Optional[float] = Field(description="Assigned score by LLM", default=0)
    explanation: Optional[str] = Field(description="Short explanation as to why the score was given", default="None") 