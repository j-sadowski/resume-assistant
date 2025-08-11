from app.config import AI_BACKEND
if AI_BACKEND == "openai":
    from app.workflow.oa_workflow import score_resume, summarize_gaps, suggest_edits
elif AI_BACKEND == "ollama":
    from .ollama_workflow import score_resume, summarize_gaps
else:
    raise ValueError(
        f"Unknown AI_BACKEND: {AI_BACKEND}. Must be 'ollama' or 'openai'.")
