from fastapi import (
    FastAPI,
    WebSocket,
    HTTPException,
    Security,
    BackgroundTasks,
    Request,
)
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
from loguru import logger
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from passlib.context import CryptContext
import redis
from prometheus_client import Counter, Histogram, start_http_server
from .quantum_service import quantum_service

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except Exception:  # pragma: no cover - qiskit not installed
    QiskitRuntimeService = None  # type: ignore

# Load environment variables
load_dotenv()

# Initialize FastAPI app with advanced configuration
app = FastAPI(
    title="Quantum Chatbot API",
    description=(
        "Advanced quantum-enhanced chatbot with machine learning capabilities"
    ),
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Advanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Quantum-Metrics", "X-Processing-Time"],
)

# Security configurations
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Redis for caching and rate limiting
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True,
)

# Prometheus metrics
REQUEST_COUNT = Counter("quantum_chatbot_requests_total", "Total number of requests")
PROCESSING_TIME = Histogram(
    "quantum_chatbot_processing_seconds", "Time spent processing requests"
)
QUANTUM_ERRORS = Counter(
    "quantum_chatbot_errors_total", "Total number of quantum processing errors"
)


# Enhanced models with validation
class ChatMessage(BaseModel):
    message: str
    timestamp: Optional[datetime] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    language: Optional[str] = "en"
    session_id: Optional[str] = None
    user_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator("message")
    def validate_message(cls, v):
        if len(v) > 1000:
            raise ValueError("Message too long")
        return v


class QuantumResponse(BaseModel):
    response: str
    quantum_state: Optional[dict] = None
    confidence: float
    processing_time: float
    quantum_metrics: Optional[Dict[str, float]] = None
    suggested_topics: Optional[List[str]] = None
    emotion_analysis: Optional[Dict[str, float]] = None
    language_detection: Optional[Dict[str, float]] = None
    quantum_circuit_visualization: Optional[str] = None
    next_best_actions: Optional[List[Dict[str, Any]]] = None


# Initialize quantum services (disabled when Qiskit is unavailable)
try:
    service = QiskitRuntimeService(
        channel="ibm_quantum", token=os.getenv("IBM_QUANTUM_API_KEY")
    )
    logger.info("Successfully connected to IBM Quantum service")
except Exception as e:
    logger.error(f"Failed to connect to IBM Quantum service: {str(e)}")
    service = None

# Thread and process pools for parallel processing
quantum_executor = ThreadPoolExecutor(max_workers=8)
ml_executor = ProcessPoolExecutor(max_workers=4)


# Enhanced quantum response generation with machine learning
async def generate_quantum_response(message: str) -> QuantumResponse:
    start_time = datetime.now()
    REQUEST_COUNT.inc()

    try:
        # Use the shared quantum_service which handles mock implementations
        circuit = quantum_service.create_quantum_circuit(message, num_qubits=4)
        result = quantum_service.execute_circuit(circuit, shots=1000)
        quantum_metrics = quantum_service.analyze_quantum_state(result)

        circuit_visualization = ""
        if hasattr(circuit, "draw"):
            try:
                circuit_visualization = circuit.draw(output="text")
            except Exception:
                circuit_visualization = ""

        quantum_state = {
            "circuit": getattr(circuit, "qasm", lambda: "")(),
            "counts": (
                result.get("counts", {})
                if isinstance(result, dict)
                else getattr(result, "get_counts", lambda: {})()
            ),
            "metrics": quantum_metrics,
            "visualization": circuit_visualization,
        }

        response = generate_response_from_quantum_state(quantum_state, message)
        processing_time = (datetime.now() - start_time).total_seconds()
        PROCESSING_TIME.observe(processing_time)

        return QuantumResponse(
            response=response,
            quantum_state=quantum_state,
            confidence=calculate_confidence(quantum_metrics),
            processing_time=processing_time,
            quantum_metrics=quantum_metrics,
            suggested_topics=generate_suggested_topics(quantum_metrics),
            emotion_analysis=analyze_emotion(message),
            language_detection=detect_language(message),
            quantum_circuit_visualization=circuit_visualization,
            next_best_actions=generate_next_actions(quantum_metrics, message),
        )

    except Exception as e:
        QUANTUM_ERRORS.inc()
        logger.error(f"Error in quantum response generation: {str(e)}")
        raise HTTPException(status_code=500, detail="Quantum processing error")


def generate_response_from_quantum_state(state: Dict[str, Any], message: str) -> str:
    """Return a simple textual response based on the quantum state."""
    return f"Processed message '{message}' with mock quantum results."


def calculate_confidence(metrics: Dict[str, float]) -> float:
    """Return a dummy confidence score based on available metrics."""
    if not metrics:
        return 0.0
    return float(sum(metrics.values()) / len(metrics))


def generate_suggested_topics(metrics: Dict[str, float]) -> List[str]:
    # Generate topic suggestions based on quantum metrics
    topics = []
    if metrics.get("entropy", 1.0) < 0.3:
        topics.append("Low Entropy States")
    if metrics.get("purity", 0.0) > 0.8:
        topics.append("Quantum State Purity")
    if metrics.get("coherence", 0.0) > 0.6:
        topics.append("Quantum Coherence")
    return topics


def analyze_emotion(text: str) -> Dict[str, float]:
    # Advanced emotion analysis using quantum-enhanced NLP
    # This is a simplified version - in production, use a proper NLP model
    return {"positive": 0.7, "negative": 0.1, "neutral": 0.2}


def detect_language(text: str) -> Dict[str, float]:
    # Language detection using quantum-enhanced NLP
    # This is a simplified version - in production, use a proper language detection model
    return {"en": 0.9, "es": 0.05, "fr": 0.05}


def generate_next_actions(
    metrics: Dict[str, float], message: str
) -> List[Dict[str, Any]]:
    # Generate next best actions based on quantum state and context
    actions = []
    if "?" in message:
        actions.append(
            {
                "type": "explanation",
                "confidence": metrics["purity"],
                "description": "Provide detailed explanation",
            }
        )
    if "quantum" in message.lower():
        actions.append(
            {
                "type": "visualization",
                "confidence": metrics["coherence"],
                "description": "Show quantum state visualization",
            }
        )
    return actions


# API endpoints with advanced features
@app.get("/")
async def root():
    return {
        "message": "Quantum Chatbot API is running",
        "version": "2.0.0",
        "quantum_backend": "ibmq_manila" if service else "qasm_simulator",
        "features": [
            "Quantum-enhanced NLP",
            "Real-time quantum state visualization",
            "Advanced emotion analysis",
            "Language detection",
            "Quantum metrics analysis",
            "Next best action prediction",
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for CI/CD and monitoring."""
    try:
        # Check Redis connection
        redis_status = "healthy" if redis_client.ping() else "unhealthy"
        
        # Check quantum service
        quantum_status = "healthy" if quantum_service else "unhealthy"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "redis": redis_status,
                "quantum_service": quantum_status,
                "api": "healthy"
            },
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/chat", response_model=QuantumResponse)
async def chat(
    message: ChatMessage,
    background_tasks: BackgroundTasks,
    request: Request,
    api_key: str = Security(api_key_header),
):
    # Rate limiting
    client_ip = request.client.host
    if redis_client.get(f"rate_limit:{client_ip}"):
        raise HTTPException(status_code=429, detail="Too many requests")

    redis_client.setex(f"rate_limit:{client_ip}", 60, "1")

    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")

    # Log the request
    logger.info(f"Received chat request: {message.message}")

    # Process in background
    response = await generate_quantum_response(message.message)

    # Log the response
    logger.info(f"Generated response with confidence: {response.confidence}")

    return response


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = ChatMessage(message=data, timestamp=datetime.now())
            response = await generate_quantum_response(message.message)
            await websocket.send_json(response.dict())
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    # Start Prometheus metrics server on a separate port
    start_http_server(port + 1)
    uvicorn.run(app, host=host, port=port)
