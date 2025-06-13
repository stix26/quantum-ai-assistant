from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from .database import Base

class Conversation(Base):
    __tablename__ = 'conversations'

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), index=True)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=True)
    quantum_state = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
