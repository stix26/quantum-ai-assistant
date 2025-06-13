from typing import Dict


class DialogManager:
    """Minimal in-memory dialog manager."""

    def __init__(self) -> None:
        self.sessions: Dict[str, Dict[str, str]] = {}

    def update_state(self, session_id: str, intent: str) -> Dict[str, str]:
        state = self.sessions.setdefault(session_id, {})
        state["last_intent"] = intent
        return state
