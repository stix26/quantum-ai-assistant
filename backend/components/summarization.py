from typing import Optional

try:
    from transformers import pipeline
    summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')
    TRANSFORMERS_AVAILABLE = True
except Exception:
    summarizer = None
    TRANSFORMERS_AVAILABLE = False


def summarize_text(text: str, max_length: int = 60, min_length: int = 10) -> str:
    """Return a short summary of the text."""
    if summarizer:
        try:
            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception:
            pass
    # Fallback: truncate text
    return ' '.join(text.split()[:max_length])
