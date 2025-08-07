
from app.config import AI_BACKEND
if AI_BACKEND == "openai":
    from app.scoring.oa_models import score_resume, summarize_gaps, suggest_edits
elif AI_BACKEND == "ollama":
    from .ollama_models import score_resume, summarize_gaps
else:
    raise ValueError(
        f"Unknown AI_BACKEND: {AI_BACKEND}. Must be 'ollama' or 'openai'.")
