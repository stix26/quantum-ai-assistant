from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from loguru import logger
from .config import settings, get_ibm_quantum_provider_config, get_backend_config

# Optional Qiskit imports. When running in environments where the heavy
# dependencies cannot be installed (e.g. during offline testing), fall back to
# lightweight mock implementations so that the rest of the application can be
# imported and tested.
QISKIT_AVAILABLE = False
if not settings.USE_QISKIT_MOCK:
    try:  # pragma: no cover - used only when Qiskit is present
        from qiskit import QuantumCircuit, execute, transpile
        from qiskit.providers.ibmq import IBMQ
        from qiskit.quantum_info import Statevector, Operator
        from qiskit.result import Result
        from qiskit.visualization import plot_bloch_multivector, plot_state_city
        from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
        from qiskit.aer import AerSimulator
        from qiskit.providers.aer.noise import NoiseModel
        from qiskit.providers.aer.noise.errors import depolarizing_error
        QISKIT_AVAILABLE = True
    except Exception as e:  # pragma: no cover - executed only without Qiskit
        logger.warning(f"Qiskit not available: {e}. Using mock classes.")
else:
    logger.info("USE_QISKIT_MOCK enabled - using mock classes")

if not QISKIT_AVAILABLE:
    class QuantumCircuit:  # type: ignore
        def __init__(self, num_qubits: int = 1):
            self.num_qubits = num_qubits
        def measure_all(self) -> None:
            pass
        def x(self, *args: Any) -> None:
            pass
        def h(self, *args: Any) -> None:
            pass
        def cx(self, *args: Any) -> None:
            pass
        def ry(self, *args: Any) -> None:
            pass

    class AerSimulator:  # type: ignore
        pass

    class NoiseModel:  # type: ignore
        pass

    class Operator:  # type: ignore
        pass

    class Result(dict):  # type: ignore
        def get_counts(self) -> Dict[str, int]:
            return {}

    def execute(*args: Any, **kwargs: Any) -> Result:  # type: ignore
        return Result()

    def transpile(circuit: QuantumCircuit, *args: Any, **kwargs: Any) -> QuantumCircuit:  # type: ignore
        return circuit

    def depolarizing_error(*args: Any, **kwargs: Any) -> None:  # type: ignore
        return None

    def complete_meas_cal(*args: Any, **kwargs: Any) -> Tuple[None, List[str]]:  # type: ignore
        return None, []

    class CompleteMeasFitter:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        class filter:  # noqa: D401 - simple wrapper
            @staticmethod
            def apply(result: Result) -> Result:
                return result

    def plot_bloch_multivector(*args: Any, **kwargs: Any) -> None:  # type: ignore
        return None

    def plot_state_city(*args: Any, **kwargs: Any) -> None:  # type: ignore
        return None

class QuantumService:
    def __init__(self):
        self.provider = None
        self.backend = None
        self.simulator = AerSimulator()
        self.initialize_provider()

    def initialize_provider(self) -> None:
        """Initialize the IBM Quantum provider with the API key."""
        try:
            if not settings.IBM_QUANTUM_API_KEY:
                logger.warning("No IBM Quantum API key provided. Using simulator only.")
                return

            IBMQ.save_account(settings.IBM_QUANTUM_API_KEY)
            self.provider = IBMQ.load_account()
            logger.info("Successfully connected to IBM Quantum")
            
            # Get the backend
            backend_config = get_backend_config()
            try:
                self.backend = self.provider.get_backend(backend_config["backend_name"])
                logger.info(f"Using backend: {self.backend.name}")
            except Exception as e:
                logger.warning(f"Could not get specified backend: {e}")
                # Fall back to least busy backend
                self.backend = self.provider.get_backend('ibmq_manila')
                logger.info(f"Falling back to backend: {self.backend.name}")

        except Exception as e:
            logger.error(f"Error initializing IBM Quantum provider: {e}")
            self.provider = None
            self.backend = None

    def create_quantum_circuit(self, text: str, num_qubits: int = 5) -> QuantumCircuit:
        """Create a quantum circuit for text encoding."""
        # Convert text to binary
        binary = ''.join(format(ord(char), '08b') for char in text)
        
        # Create circuit
        circuit = QuantumCircuit(num_qubits)
        
        # Initial encoding
        for i, bit in enumerate(binary[:num_qubits]):
            if bit == '1':
                circuit.x(i)
        
        # Apply Hadamard gates for superposition
        for i in range(num_qubits):
            circuit.h(i)
        
        # Apply CNOT gates for entanglement
        for i in range(num_qubits - 1):
            circuit.cx(i, i + 1)
        
        # Apply rotation gates based on text
        for i in range(num_qubits):
            angle = (ord(text[i % len(text)]) / 255) * np.pi
            circuit.ry(angle, i)
        
        # Add measurement
        circuit.measure_all()
        
        return circuit

    def execute_circuit(self, circuit: QuantumCircuit, shots: int = 1000) -> Result:
        """Execute a quantum circuit on the available backend."""
        try:
            if self.backend:
                # Transpile the circuit for the backend
                transpiled_circuit = transpile(
                    circuit,
                    self.backend,
                    optimization_level=settings.OPTIMIZATION_LEVEL,
                    layout_method=settings.LAYOUT_METHOD
                )
                
                # Execute on real quantum computer
                job = execute(
                    transpiled_circuit,
                    self.backend,
                    shots=shots,
                    optimization_level=settings.OPTIMIZATION_LEVEL
                )
                
                # Get results
                result = job.result()
                
                # Apply error mitigation if enabled
                if settings.USE_ERROR_MITIGATION:
                    result = self.apply_error_mitigation(result, transpiled_circuit)
                
                return result
            else:
                # Fall back to simulator
                logger.info("Using simulator as fallback")
                job = execute(circuit, self.simulator, shots=shots)
                return job.result() if hasattr(job, "result") else job
                
        except Exception as e:
            logger.error(f"Error executing circuit: {e}")
            # Fall back to simulator
            job = execute(circuit, self.simulator, shots=shots)
            return job.result() if hasattr(job, "result") else job

    def apply_error_mitigation(self, result: Result, circuit: QuantumCircuit) -> Result:
        """Apply error mitigation to the results."""
        try:
            if settings.ERROR_MITIGATION_METHOD == "zne":
                # Zero-noise extrapolation
                noise_model = NoiseModel()
                noise_model.add_all_qubit_quantum_error(
                    depolarizing_error(0.1, 1), ['u1', 'u2', 'u3']
                )
                
                # Execute with different noise levels
                noise_levels = [0, 0.1, 0.2]
                results = []
                
                for noise_level in noise_levels:
                    noise_model.add_all_qubit_quantum_error(
                        depolarizing_error(noise_level, 1), ['u1', 'u2', 'u3']
                    )
                    result_noisy = execute(
                        circuit,
                        self.simulator,
                        noise_model=noise_model,
                        shots=1000
                    ).result()
                    results.append(result_noisy)
                
                # Extrapolate to zero noise
                mitigated_result = self._extrapolate_zero_noise(results, noise_levels)
                return mitigated_result
                
            elif settings.ERROR_MITIGATION_METHOD == "measurement":
                # Measurement error mitigation
                meas_calibs, state_labels = complete_meas_cal(
                    qubit_list=range(circuit.num_qubits),
                    circlabel='mcal'
                )
                
                # Execute calibration circuits
                cal_results = execute(
                    meas_calibs,
                    self.backend,
                    shots=1000
                ).result()
                
                # Create measurement filter
                meas_fitter = CompleteMeasFitter(cal_results, state_labels)
                
                # Apply correction
                mitigated_result = meas_fitter.filter.apply(result)
                return mitigated_result
                
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error applying error mitigation: {e}")
            return result

    def _extrapolate_zero_noise(self, results: List[Result], noise_levels: List[float]) -> Result:
        """Extrapolate results to zero noise."""
        # Implement zero-noise extrapolation
        # This is a simplified version - in practice, you'd want more sophisticated extrapolation
        return results[0]  # For now, just return the noiseless result

    def analyze_quantum_state(self, result: Result) -> Dict[str, Any]:
        """Analyze the quantum state and return metrics."""
        counts = result.get_counts()
        total_shots = sum(counts.values())

        if total_shots == 0:
            return {
                "counts": counts,
                "probabilities": {},
                "entropy": 0.0,
                "purity": 0.0,
                "coherence": 0.0,
                "most_probable_state": None,
                "most_probable_probability": 0.0,
            }

        # Calculate probabilities
        probabilities = {state: count / total_shots for state, count in counts.items()}
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probabilities.values())
        
        # Calculate purity
        purity = sum(p**2 for p in probabilities.values())
        
        # Calculate coherence
        coherence = sum(abs(p) for p in probabilities.values())
        
        # Get most probable state
        most_probable_state = max(probabilities.items(), key=lambda x: x[1])
        
        return {
            "counts": counts,
            "probabilities": probabilities,
            "entropy": entropy,
            "purity": purity,
            "coherence": coherence,
            "most_probable_state": most_probable_state[0],
            "most_probable_probability": most_probable_state[1]
        }

    def visualize_quantum_state(self, result: Result) -> Dict[str, str]:
        """Generate visualizations of the quantum state."""
        visualizations = {}
        
        try:
            # Bloch sphere visualization
            statevector = Statevector.from_instruction(result.get_unitary())
            bloch_fig = plot_bloch_multivector(statevector)
            bloch_buffer = io.BytesIO()
            bloch_fig.savefig(bloch_buffer, format='png')
            bloch_buffer.seek(0)
            visualizations['bloch'] = base64.b64encode(bloch_buffer.getvalue()).decode()
            
            # State city plot
            city_fig = plot_state_city(statevector)
            city_buffer = io.BytesIO()
            city_fig.savefig(city_buffer, format='png')
            city_buffer.seek(0)
            visualizations['city'] = base64.b64encode(city_buffer.getvalue()).decode()
            
            plt.close('all')
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return visualizations

    def get_available_backends(self) -> List[Dict[str, Any]]:
        """Get information about available quantum backends."""
        backends = []
        
        try:
            if self.provider:
                for backend in self.provider.backends():
                    if backend.status().operational:
                        backends.append({
                            "name": backend.name(),
                            "status": backend.status().status_msg,
                            "qubits": backend.configuration().n_qubits,
                            "pending_jobs": backend.status().pending_jobs,
                            "is_simulator": backend.configuration().simulator
                        })
        except Exception as e:
            logger.error(f"Error getting available backends: {e}")
        
        return backends

    def get_backend_status(self) -> Dict[str, Any]:
        """Get the status of the current backend."""
        if not self.backend:
            return {"status": "No backend available"}
            
        try:
            status = self.backend.status()
            return {
                "name": self.backend.name(),
                "status": status.status_msg,
                "pending_jobs": status.pending_jobs,
                "operational": status.operational,
                "qubits": self.backend.configuration().n_qubits,
                "is_simulator": self.backend.configuration().simulator
            }
        except Exception as e:
            logger.error(f"Error getting backend status: {e}")
            return {"status": "Error getting status"}

# Create a singleton instance
quantum_service = QuantumService() 