from backend.config import get_backend_config, settings

def test_default_backend_settings():
    cfg = get_backend_config()
    assert cfg["backend_name"] == settings.DEFAULT_BACKEND
    assert cfg["shots"] == settings.MAX_SHOTS
