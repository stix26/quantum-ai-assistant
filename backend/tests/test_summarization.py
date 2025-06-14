import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.summarization import summarize_text

def test_summarize_text():
    text = "This is a very long text that should be summarized into a shorter form. " * 5
    summary = summarize_text(text)
    assert isinstance(summary, str)
    assert len(summary) > 0
