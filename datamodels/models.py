from typing import Optional
from pydantic import BaseModel, Field


class ComparisonExtract(BaseModel):
    is_valid: bool = Field(description="Whether this text is describing terms for comparing a resume against a job description")
    confidence: float = Field(description="Confidence score that this is a request")
    rationale: str = Field(description="A concise explanation on why the prompt was given this score")

class WorkflowReqs(BaseModel):
    resume: Optional[str] = Field(description="The plain text of a resume")
    keywords: str = Field(description="The job title")
    city: str = Field(description="The city where the job is located. DO NOT include the state")
    limit: Optional[int] = Field(description="The maximum number of items to return", default=20)
    hybrid: Optional[bool] = Field(description="Whether to apply a hybrid job-only filter", default=False)

class JDScore(BaseModel):
    """Score the JD against the resume"""
    score: float = Field(description="Resume suitability score")
    explanation: str = Field(description="Explanation of suitability score")

class ResumeDigest(BaseModel):
    """Summarize the resume"""
    summary: str = Field(description="Summary of the input resume")