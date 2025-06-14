import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from sqlalchemy import text
from database import engine

@pytest.mark.asyncio
async def test_database_connection():
    async with engine.connect() as conn:
        result = await conn.execute(text('SELECT 1'))
        assert result.scalar() == 1
