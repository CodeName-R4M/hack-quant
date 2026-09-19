"""
Tests for IBM Quantum Integration and Aer Simulator Execution
"""

import pytest
from traffic_quantum.quantum.ibm_qpu import IBMQuantumManager


def test_ibm_quantum_manager_initialization():
    mgr = IBMQuantumManager()
    assert isinstance(mgr.status, dict)
    assert "authenticated" in mgr.status
    assert "message" in mgr.status


def test_ibm_quantum_aer_circuit_execution():
    mgr = IBMQuantumManager()
    # Simple 2-qubit QUBO
    Q = {(0, 0): -2.0, (1, 1): -2.0, (0, 1): 3.0}
    result = mgr.run_qaoa_on_qiskit(Q, n_qubits=2, shots=256)

    assert result["status"] == "COMPLETED"
    assert "job_id" in result
    assert result["n_qubits"] == 2
    assert result["shots"] == 256
    assert len(result["top_bitstring"]) == 2
    assert "counts" in result
