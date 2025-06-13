from typing import Dict, List


def analyze_intent_entities(text: str) -> Dict[str, List[str]]:
    """Simple intent and entity extraction placeholder."""
    intent = "question" if text.strip().endswith("?") else "statement"
    entities: List[str] = []
    return {"intent": intent, "entities": entities}
