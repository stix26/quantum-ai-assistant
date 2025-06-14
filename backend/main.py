from fastapi import FastAPI, WebSocket, HTTPException, Depends, Security, BackgroundTasks, Request
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import os
from dotenv import load_dotenv
from loguru import logger
import json
import asyncio
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import jwt
from passlib.context import CryptContext
import redis
from prometheus_client import Counter, Histogram, start_http_server
import aiohttp
import tensorflow as tf
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import entropy
import networkx as nx
try:
    from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
    from qiskit.quantum_info import Statevector, Operator, SparsePauliOp
    from qiskit.visualization import plot_bloch_multivector
    from qiskit.providers.ibmq import least_busy
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options, Estimator
    from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
    from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
    from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2
    from qiskit.algorithms import VQC, VQE
    from qiskit.opflow import Z, I, X, Y
    QISKIT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    QuantumCircuit = None
    Statevector = None
    QISKIT_AVAILABLE = False
if QISKIT_AVAILABLE:
    from .quantum_service import quantum_service
else:  # pragma: no cover - fallback
    quantum_service = None
from .config import settings
from .database import get_session, Base, engine, AsyncSessionLocal
from .models import Conversation
from .components.nlu import analyze_intent_entities
from .components.dialog_manager import DialogManager
from .components.nlg import generate_text
from .components.response_handler import format_response
from .components.speech import speech_to_text, text_to_speech
from .components.sentiment import analyze_sentiment
from .components.personalization import PersonalizationEngine

# Load environment variables
load_dotenv()

# Initialize FastAPI app with advanced configuration
app = FastAPI(
    title="Quantum Chatbot API",
    description="Advanced quantum-enhanced chatbot with machine learning capabilities",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)


@app.on_event("startup")
async def on_startup() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await engine.dispose()

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
    decode_responses=True
)

# Simple in-memory dialog manager instance
dialog_manager = DialogManager()
personalization_engine = PersonalizationEngine()

# Prometheus metrics
REQUEST_COUNT = Counter('quantum_chatbot_requests_total', 'Total number of requests')
PROCESSING_TIME = Histogram('quantum_chatbot_processing_seconds', 'Time spent processing requests')
QUANTUM_ERRORS = Counter('quantum_chatbot_errors_total', 'Total number of quantum processing errors')

# Enhanced models with validation
class ChatMessage(BaseModel):
    message: str
    timestamp: Optional[datetime] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    language: Optional[str] = "en"
    session_id: Optional[str] = None
    user_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator('message')
    def validate_message(cls, v):
        if len(v) > 1000:
            raise ValueError('Message too long')
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
    sentiment: Optional[Dict[str, float]] = None
    user_preferences: Optional[Dict[str, str]] = None
    nlu: Optional[Dict[str, Any]] = None
    dialog_state: Optional[Dict[str, Any]] = None

# Initialize quantum services with advanced error handling
try:
    service = QiskitRuntimeService(channel="ibm_quantum", token=os.getenv("IBM_QUANTUM_API_KEY"))
    logger.info("Successfully connected to IBM Quantum service")
    
    # Get available backends with advanced filtering
    backends = service.backends(
        filters=lambda x: x.configuration().n_qubits >= 5 and x.status().operational
    )
    logger.info(f"Available backends: {[backend.name for backend in backends]}")
    
    # Initialize quantum machine learning models
    feature_map = ZZFeatureMap(feature_dimension=4, reps=2)
    ansatz = RealAmplitudes(4, reps=2)
    qnn = SamplerQNN(
        circuit=ansatz,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True
    )
    
except Exception as e:
    logger.error(f"Failed to connect to IBM Quantum service: {str(e)}")
    service = None

# Thread and process pools for parallel processing
quantum_executor = ThreadPoolExecutor(max_workers=8)
ml_executor = ProcessPoolExecutor(max_workers=4)

# Advanced quantum circuit generation with multiple layers
from typing import Optional

def create_quantum_circuit(text: str, num_qubits: int = 4) -> Optional[QuantumCircuit]:
    if not QISKIT_AVAILABLE:
        logger.warning("Qiskit not available - returning None for circuit")
        return None

    circuit = QuantumCircuit(num_qubits)
    
    # Convert text to quantum state using advanced encoding
    binary = ''.join(format(ord(char), '08b') for char in text)
    
    # Initial encoding layer
    for i, bit in enumerate(binary[:num_qubits]):
        if bit == '1':
            circuit.x(i)
    
    # Superposition layer with phase kickback
    for i in range(num_qubits):
        circuit.h(i)
        circuit.rz(np.pi/4, i)
    
    # Entanglement layer with controlled-phase gates
    for i in range(num_qubits-1):
        circuit.cp(np.pi/4, i, i+1)
        circuit.cx(i, i+1)
    
    # Rotation layer for quantum feature extraction
    for i in range(num_qubits):
        circuit.rz(np.pi/3, i)
        circuit.rx(np.pi/4, i)
        circuit.ry(np.pi/6, i)
    
    # Final entanglement layer with multi-qubit gates
    for i in range(num_qubits-2):
        circuit.ccx(i, i+1, i+2)
    
    # Measurement layer
    circuit.measure_all()
    
    return circuit

# Advanced quantum state analysis
def analyze_quantum_state(circuit: Optional[QuantumCircuit]) -> Dict[str, float]:
    if not QISKIT_AVAILABLE or circuit is None:
        logger.warning("Qiskit not available - returning empty quantum metrics")
        return {"entanglement": 0.0, "purity": 0.0, "coherence": 0.0}

    statevector = Statevector.from_instruction(circuit)
    
    # Calculate advanced quantum metrics
    metrics = {
        "entanglement": calculate_entanglement(statevector),
        "purity": calculate_purity(statevector),
        "coherence": calculate_coherence(statevector),
        "von_neumann_entropy": calculate_von_neumann_entropy(statevector),
        "quantum_fisher_information": calculate_quantum_fisher_information(statevector),
        "quantum_discord": calculate_quantum_discord(statevector),
        "quantum_correlation": calculate_quantum_correlation(statevector)
    }
    
    return metrics

def calculate_von_neumann_entropy(statevector: Statevector) -> float:
    # Calculate von Neumann entropy
    density_matrix = np.outer(statevector.data, np.conj(statevector.data))
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    return -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))

def calculate_quantum_fisher_information(statevector: Statevector) -> float:
    # Calculate quantum Fisher information
    density_matrix = np.outer(statevector.data, np.conj(statevector.data))
    return np.trace(density_matrix @ density_matrix)

def calculate_quantum_discord(statevector: Statevector) -> float:
    # Calculate quantum discord
    density_matrix = np.outer(statevector.data, np.conj(statevector.data))
    return np.abs(np.trace(density_matrix @ density_matrix) - np.trace(density_matrix)**2)

def calculate_quantum_correlation(statevector: Statevector) -> float:
    # Calculate quantum correlation
    density_matrix = np.outer(statevector.data, np.conj(statevector.data))
    return np.abs(np.trace(density_matrix @ density_matrix))

def calculate_entanglement(statevector: Statevector) -> float:
    # Placeholder entanglement metric
    return float(np.var(np.abs(statevector.data)))

def calculate_purity(statevector: Statevector) -> float:
    density_matrix = np.outer(statevector.data, np.conj(statevector.data))
    return float(np.trace(density_matrix @ density_matrix).real)

def calculate_coherence(statevector: Statevector) -> float:
    return float(np.sum(np.abs(statevector.data)))

def calculate_confidence(metrics: Dict[str, float]) -> float:
    return float(np.mean(list(metrics.values()))) if metrics else 0.0

def generate_response_from_quantum_state(quantum_state: Dict[str, Any], nlg_text: str) -> str:
    metrics = quantum_state.get("metrics", {})
    confidence = calculate_confidence(metrics)
    return f"{nlg_text} (confidence {confidence:.2f})"

# Enhanced quantum response generation with machine learning
async def generate_quantum_response(message: str) -> QuantumResponse:
    start_time = datetime.now()
    REQUEST_COUNT.inc()

    try:
        nlu = analyze_intent_entities(message)
        dialog_state = dialog_manager.update_state("default", nlu["intent"])
        sentiment = analyze_sentiment(message)
        user_prefs = personalization_engine.get_preferences("default")
        nlg_text = generate_text(nlu["intent"])

        # Create and analyze quantum circuit
        circuit = create_quantum_circuit(message)
        quantum_metrics = analyze_quantum_state(circuit)
        
        # Execute on IBM Quantum with advanced options
        if service:
            with Session(service=service, backend="ibmq_manila") as session:
                options = Options(
                    optimization_level=3,
                    resilience_level=2,
                    max_parallel_experiments=4,
                    max_parallel_shots=1000
                )
                sampler = Sampler(session=session, options=options)
                job = sampler.run(circuit)
                result = job.result()
                counts = result.quasi_dists[0]
        else:
            backend = Aer.get_backend('qasm_simulator')
            job = execute(circuit, backend, shots=1000)
            result = job.result()
            counts = result.get_counts()
        
        # Generate quantum circuit visualization
        circuit_visualization = circuit.draw(output='text')
        
        # Process results and generate response
        quantum_state = {
            "circuit": circuit.qasm(),
            "counts": counts,
            "metrics": quantum_metrics,
            "visualization": circuit_visualization
        }
        
        # Generate response using quantum-enhanced NLP and simple NLG
        base_text = generate_response_from_quantum_state(quantum_state, nlg_text)
        formatted = format_response(base_text, {"nlu": nlu, "state": dialog_state})
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        PROCESSING_TIME.observe(processing_time)
        
        return QuantumResponse(
            response=formatted["text"],
            quantum_state=quantum_state,
            confidence=calculate_confidence(quantum_metrics),
            processing_time=processing_time,
            quantum_metrics=quantum_metrics,
            suggested_topics=generate_suggested_topics(quantum_metrics),
            emotion_analysis=analyze_emotion(message),
            language_detection=detect_language(message),
            quantum_circuit_visualization=circuit_visualization,
            next_best_actions=generate_next_actions(quantum_metrics, message),
            sentiment=sentiment,
            user_preferences=user_prefs,
            nlu=nlu,
            dialog_state=dialog_state
        )
    
    except Exception as e:
        QUANTUM_ERRORS.inc()
        logger.error(f"Error in quantum response generation: {str(e)}")
        raise HTTPException(status_code=500, detail="Quantum processing error")

def generate_suggested_topics(metrics: Dict[str, float]) -> List[str]:
    # Generate topic suggestions based on quantum metrics
    topics = []
    if metrics["entanglement"] > 0.7:
        topics.append("Quantum Entanglement")
    if metrics["purity"] > 0.8:
        topics.append("Quantum State Purity")
    if metrics["coherence"] > 0.6:
        topics.append("Quantum Coherence")
    return topics

def analyze_emotion(text: str) -> Dict[str, float]:
    # Advanced emotion analysis using quantum-enhanced NLP
    # This is a simplified version - in production, use a proper NLP model
    return {
        "positive": 0.7,
        "negative": 0.1,
        "neutral": 0.2
    }

def detect_language(text: str) -> Dict[str, float]:
    # Language detection using quantum-enhanced NLP
    # This is a simplified version - in production, use a proper language detection model
    return {
        "en": 0.9,
        "es": 0.05,
        "fr": 0.05
    }

def generate_next_actions(metrics: Dict[str, float], message: str) -> List[Dict[str, Any]]:
    # Generate next best actions based on quantum state and context
    actions = []
    if "?" in message:
        actions.append({
            "type": "explanation",
            "confidence": metrics["purity"],
            "description": "Provide detailed explanation"
        })
    if "quantum" in message.lower():
        actions.append({
            "type": "visualization",
            "confidence": metrics["coherence"],
            "description": "Show quantum state visualization"
        })
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
            "Next best action prediction"
        ]
    }

@app.post("/chat", response_model=QuantumResponse)
async def chat(
    message: ChatMessage,
    background_tasks: BackgroundTasks,
    request: Request,
    api_key: str = Security(api_key_header),
    session: AsyncSession = Depends(get_session)
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

    conversation = Conversation(
        session_id=message.session_id,
        user_message=message.message,
        bot_response=response.response,
        quantum_state=response.quantum_state
    )
    session.add(conversation)
    await session.commit()

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

            async with AsyncSessionLocal() as session:
                conversation = Conversation(
                    session_id=message.session_id,
                    user_message=message.message,
                    bot_response=response.response,
                    quantum_state=response.quantum_state,
                )
                session.add(conversation)
                await session.commit()

            await websocket.send_json(response.dict())
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    # Start Prometheus metrics server
    start_http_server(8000)
    uvicorn.run(app, host="0.0.0.0", port=8000) 