from backend import quantum_service
from backend.quantum_service import QuantumService


def test_provider_none_without_api_key(monkeypatch):
    monkeypatch.setattr(quantum_service.settings, "IBM_QUANTUM_API_KEY", "")
    service = QuantumService()
    assert service.provider is None  # nosec B101


def test_create_circuit():
    service = QuantumService()
    circuit = service.create_quantum_circuit("hi", num_qubits=2)
    assert getattr(circuit, "num_qubits", 0) == 2  # nosec B101
