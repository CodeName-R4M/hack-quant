"""
IBM Quantum Hardware Integration Module
Connects to IBM Quantum Platform (Qiskit Runtime) to run QAOA traffic optimization
circuits on real superconducting quantum hardware (e.g. 127-qubit Eagle QPUs)
or high-fidelity Qiskit Aer simulators.
"""

from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# Load environment variables if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class IBMQuantumManager:
    """Manages authentication, backend selection, and circuit execution on IBM Quantum."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("IBM_QUANTUM_TOKEN", "")
        self.service = None
        self.status: Dict[str, Any] = {
            "authenticated": False,
            "channel": None,
            "backends": [],
            "message": "Not connected",
            "active_backend": None,
        }
        if self.token:
            self.connect()

    def connect(self) -> Dict[str, Any]:
        """Attempt authentication via IBM Cloud or IBM Quantum Platform."""
        if not self.token:
            self.status = {
                "authenticated": False,
                "channel": None,
                "backends": [],
                "message": "No IBM Quantum API token provided in .env (IBM_QUANTUM_TOKEN).",
                "active_backend": None,
            }
            return self.status

        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            
            # Try ibm_cloud channel first (standard for IAM keys)
            service = None
            channel = None
            err_msg = ""
            
            try:
                service = QiskitRuntimeService(channel="ibm_cloud", token=self.token)
                channel = "ibm_cloud"
            except Exception as e_cloud:
                err_msg = str(e_cloud)
                try:
                    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=self.token)
                    channel = "ibm_quantum_platform"
                except Exception as e_platform:
                    err_msg = f"Cloud: {e_cloud}; Platform: {e_platform}"

            if service:
                self.service = service
                backends = [b.name for b in service.backends()]
                self.status = {
                    "authenticated": True,
                    "channel": channel,
                    "backends": backends,
                    "message": f"Successfully connected to IBM Quantum via {channel} ({len(backends)} backends available).",
                    "active_backend": backends[0] if backends else None,
                }
                return self.status
            else:
                # Provide clear instruction if no cloud instance was provisioned yet
                if "No matching instances found" in err_msg:
                    msg = (
                        "IBM Cloud IAM API key is valid, but no 'Qiskit Runtime' service instance "
                        "has been provisioned in your IBM Cloud account yet. "
                        "Provision a free/Lite instance at https://cloud.ibm.com/catalog/services/qiskit-runtime "
                        "or copy your token from https://quantum.ibm.com/account."
                    )
                else:
                    msg = f"Authentication failed: {err_msg}"

                self.status = {
                    "authenticated": False,
                    "channel": None,
                    "backends": [],
                    "message": msg,
                    "active_backend": None,
                }
                return self.status

        except ImportError as e:
            self.status = {
                "authenticated": False,
                "channel": None,
                "backends": [],
                "message": f"Qiskit IBM Runtime library not installed: {e}",
                "active_backend": None,
            }
            return self.status
        except Exception as e:
            self.status = {
                "authenticated": False,
                "channel": None,
                "backends": [],
                "message": f"Connection error: {str(e)}",
                "active_backend": None,
            }
            return self.status

    def run_qaoa_on_qiskit(
        self,
        Q: Dict[Tuple[int, int], float],
        n_qubits: int = 6,
        shots: int = 1024,
        backend_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the QAOA traffic optimization circuit using Qiskit.
        If real IBM QPU is connected and requested, dispatches to IBM Quantum Runtime;
        otherwise runs on high-speed local Qiskit Aer simulator with full noise/circuit statistics.
        """
        from qiskit import QuantumCircuit
        import qiskit_aer

        # Map QUBO to Ising couplings
        # H_C = sum J_ij Z_i Z_j + sum h_i Z_i
        h: Dict[int, float] = {}
        J: Dict[Tuple[int, int], float] = {}
        for i in range(n_qubits):
            h[i] = 0.5 * Q.get((i, i), 0.0)
            for j in range(n_qubits):
                if i != j:
                    h[i] += 0.25 * (Q.get((i, j), 0.0) + Q.get((j, i), 0.0))

        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                coeff = 0.25 * (Q.get((i, j), 0.0) + Q.get((j, i), 0.0))
                if abs(coeff) > 1e-6:
                    J[(i, j)] = coeff

        # Build p=1 QAOA ansatz circuit in Qiskit
        gamma = 0.392
        beta = 0.785

        qc = QuantumCircuit(n_qubits, n_qubits)
        # Initial state: |+>^(x n)
        for q in range(n_qubits):
            qc.h(q)

        # Cost unitary U(C, gamma)
        for i, hi in h.items():
            if abs(hi) > 1e-6:
                qc.rz(2.0 * gamma * hi, i)

        for (i, j), Jij in J.items():
            qc.cx(i, j)
            qc.rz(2.0 * gamma * Jij, j)
            qc.cx(i, j)

        # Mixer unitary U(B, beta)
        for q in range(n_qubits):
            qc.rx(2.0 * beta, q)

        # Measurement
        qc.measure(range(n_qubits), range(n_qubits))

        # Check if running on real QPU
        if self.service and backend_name and backend_name in self.status.get("backends", []):
            try:
                backend = self.service.backend(backend_name)
                job = backend.run(qc, shots=shots)
                return {
                    "mode": "ibm_hardware_qpu",
                    "backend": backend_name,
                    "job_id": job.job_id(),
                    "status": "QUEUED_ON_QPU",
                    "shots": shots,
                    "circuit_depth": qc.depth(),
                    "n_qubits": n_qubits,
                    "portal_url": f"https://quantum.ibm.com/jobs/{job.job_id()}",
                }
            except Exception as e:
                pass  # Fallback to Aer

        # Local Qiskit Aer simulation
        simulator = qiskit_aer.AerSimulator()
        start_time = time.time()
        job = simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        duration_ms = (time.time() - start_time) * 1000.0

        # Sort bitstrings by frequency
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        best_bitstring_str = sorted_counts[0][0]
        best_bitstring = [int(b) for b in reversed(best_bitstring_str)]

        return {
            "mode": "qiskit_aer_simulator",
            "backend": "qiskit_aer_simulator",
            "job_id": f"aer-sim-{int(time.time() * 1000)}",
            "status": "COMPLETED",
            "execution_time_ms": round(duration_ms, 2),
            "shots": shots,
            "circuit_depth": qc.depth(),
            "n_qubits": n_qubits,
            "top_bitstring": best_bitstring,
            "top_bitstring_str": best_bitstring_str,
            "top_frequency": sorted_counts[0][1],
            "counts": dict(sorted_counts[:8]),
        }
