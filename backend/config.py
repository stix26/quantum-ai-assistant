import os
from dotenv import load_dotenv
from typing import Dict, Any

try:  # Pydantic <2.2
    from pydantic import BaseSettings
except Exception:  # pragma: no cover - for Pydantic >=2.2
    from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    # IBM Quantum settings
    IBM_QUANTUM_API_KEY: str = os.getenv("IBM_QUANTUM_API_KEY", "")
    IBM_QUANTUM_HUB: str = os.getenv("IBM_QUANTUM_HUB", "ibm-q")
    IBM_QUANTUM_GROUP: str = os.getenv("IBM_QUANTUM_GROUP", "open")
    IBM_QUANTUM_PROJECT: str = os.getenv("IBM_QUANTUM_PROJECT", "main")

    # Default quantum backend settings
    DEFAULT_BACKEND: str = "ibmq_manila"  # Default to a 5-qubit system
    MAX_SHOTS: int = 1000
    MAX_EXPERIMENTS: int = 300

    # Circuit optimization settings
    OPTIMIZATION_LEVEL: int = 3  # Maximum optimization
    LAYOUT_METHOD: str = "sabre"  # Best layout method for most cases

    # Error mitigation settings
    USE_ERROR_MITIGATION: bool = True
    ERROR_MITIGATION_METHOD: str = "zne"  # Zero-noise extrapolation

    # Runtime settings
    RUNTIME_MAX_EXECUTION_TIME: int = 300  # 5 minutes
    RUNTIME_MAX_SHOTS: int = 1000

    # Cache settings
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # 1 hour

    # API settings
    API_KEY: str = os.getenv("API_KEY", "")
    API_KEY_HEADER: str = "X-API-Key"

    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", os.urandom(32).hex())
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"

    # Offline/testing settings
    USE_QISKIT_MOCK: bool = os.getenv("USE_QISKIT_MOCK", "false").lower() in (
        "1",
        "true",
    )

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


# IBM Quantum provider configuration
def get_ibm_quantum_provider_config() -> Dict[str, Any]:
    return {
        "hub": settings.IBM_QUANTUM_HUB,
        "group": settings.IBM_QUANTUM_GROUP,
        "project": settings.IBM_QUANTUM_PROJECT,
    }


# Backend configuration
def get_backend_config() -> Dict[str, Any]:
    return {
        "backend_name": settings.DEFAULT_BACKEND,
        "shots": settings.MAX_SHOTS,
        "max_experiments": settings.MAX_EXPERIMENTS,
        "optimization_level": settings.OPTIMIZATION_LEVEL,
        "layout_method": settings.LAYOUT_METHOD,
        "use_error_mitigation": settings.USE_ERROR_MITIGATION,
        "error_mitigation_method": settings.ERROR_MITIGATION_METHOD,
    }


# Runtime configuration
def get_runtime_config() -> Dict[str, Any]:
    return {
        "max_execution_time": settings.RUNTIME_MAX_EXECUTION_TIME,
        "max_shots": settings.RUNTIME_MAX_SHOTS,
    }


# Cache configuration
def get_cache_config() -> Dict[str, Any]:
    return {
        "enabled": settings.CACHE_ENABLED,
        "ttl": settings.CACHE_TTL,
    }
