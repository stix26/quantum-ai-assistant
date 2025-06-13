from typing import Any, Dict


def format_response(text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Combine text with metadata for returning to the UI."""
    return {"text": text, "metadata": metadata}
