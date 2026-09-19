"""Hybrid Signal Controller.

Coordinates quantum optimization (or exact brute-force verification) for phase selection
with classical adaptive timing for green phase durations:
    duration_i = clamp(base + k * queue_active, min_green, max_green)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from traffic_quantum.config import DEFAULT_CONFIG, MasterConfig
from traffic_quantum.controllers.base import BaseController
from traffic_quantum.network import RoadNetwork
from traffic_quantum.quantum.qubo import TrafficQUBOBuilder
from traffic_quantum.quantum.brute_force import BruteForceOptimizer


class HybridController(BaseController):
    """Hybrid controller combining QUBO/QAOA phase selection with classical green timing."""

    def __init__(
        self,
        network: RoadNetwork,
        config: Optional[MasterConfig] = None,
        solver_mode: str = "brute_force",  # 'brute_force' or 'qaoa'
    ):
        super().__init__(network, config)
        self.solver_mode = solver_mode
        self.name = f"Hybrid Controller ({'QAOA' if solver_mode == 'qaoa' else 'Brute-Force'})"
        
        self.qubo_builder = TrafficQUBOBuilder(network, self.config)
        self.qaoa_solver = None  # Lazily instantiated when needed
        
        # State tracking per intersection
        self.current_phases: Dict[int, int] = {
            node: self.config.simulation.phase_ns_green
            for node in self.network.graph.nodes
        }
        self.phase_end_ticks: Dict[int, int] = {
            node: 0 for node in self.network.graph.nodes
        }
        self.last_reopt_tick: int = -999

        # Metrics & Optimization Logs
        self.optimization_history: List[Dict[str, object]] = []

    def set_solver_mode(self, mode: str) -> None:
        """Switches between 'brute_force' and 'qaoa' optimization backends."""
        if mode not in ("brute_force", "qaoa"):
            raise ValueError("solver_mode must be 'brute_force' or 'qaoa'")
        self.solver_mode = mode
        self.name = f"Hybrid Controller ({'QAOA' if mode == 'qaoa' else 'Brute-Force'})"

    def reset(self) -> None:
        """Resets controller state."""
        self.current_phases = {
            node: self.config.simulation.phase_ns_green
            for node in self.network.graph.nodes
        }
        self.phase_end_ticks = {
            node: 0 for node in self.network.graph.nodes
        }
        self.last_reopt_tick = -999
        self.optimization_history = []
        if self.qaoa_solver is not None:
            self.qaoa_solver.reset_cache()

    def _compute_classical_duration(self, node: int, phase: int, simulator) -> int:
        """Classical logic: sets green duration based on active queue length.
        
        Formula: clamp(base + k * queue_active, min_green, max_green)
        """
        q_lens = simulator.get_approach_queue_lengths(node)
        if phase == self.config.simulation.phase_ns_green:
            q_active = q_lens.get("N", 0) + q_lens.get("S", 0)
        else:
            q_active = q_lens.get("E", 0) + q_lens.get("W", 0)

        cfg = self.config.hybrid
        raw_dur = cfg.base_green_sec + cfg.k_queue * q_active
        clamped_dur = max(cfg.min_green_sec, min(int(round(raw_dur)), cfg.max_green_sec))
        return clamped_dur

    def get_phases(
        self,
        current_tick: int,
        simulator,
        emergency_biases: Optional[Dict[int, str]] = None,
    ) -> Dict[int, int]:
        """Calculates signal phases. Re-optimizes every reopt_interval_sec or when durations expire."""
        reopt_interval = self.config.hybrid.reopt_interval_sec
        needs_reopt = (current_tick - self.last_reopt_tick >= reopt_interval) or any(
            current_tick >= self.phase_end_ticks[node] for node in self.network.graph.nodes
        )

        if needs_reopt:
            # 1. Build QUBO from network state
            Q, C0 = self.qubo_builder.build_qubo(simulator, emergency_biases=emergency_biases)

            # 2. Solve QUBO (Quantum or Brute-Force verification)
            best_x, opt_metadata = self._solve_qubo(Q, C0)

            # 3. Apply phase choices and compute classical green durations
            for node_id in range(self.network.num_intersections):
                phase = int(best_x[node_id])
                self.current_phases[node_id] = phase
                duration = self._compute_classical_duration(node_id, phase, simulator)
                self.phase_end_ticks[node_id] = current_tick + duration

            self.last_reopt_tick = current_tick
            opt_metadata["tick"] = current_tick
            opt_metadata["applied_phases"] = dict(self.current_phases)
            self.optimization_history.append(opt_metadata)

        return dict(self.current_phases)

    def _solve_qubo(self, Q: np.ndarray, C0: float) -> Tuple[np.ndarray, Dict[str, object]]:
        """Invokes the selected optimization backend."""
        # Exact brute-force optimum is always computed for baseline/comparison logging
        exact_x, exact_cost, _ = BruteForceOptimizer.solve(Q, C0)

        if self.solver_mode == "brute_force":
            return exact_x, {
                "solver": "brute_force",
                "best_cost": exact_cost,
                "exact_cost": exact_cost,
                "approximation_ratio": 1.0,
                "found_exact_optimum": True,
                "bitstring": "".join(map(str, exact_x)),
            }
        
        # QAOA Solver backend (Phase 4 integration)
        if self.qaoa_solver is None:
            from traffic_quantum.quantum.qaoa import QAOATrafficSolver
            self.qaoa_solver = QAOATrafficSolver(self.network.num_intersections, self.config)

        qaoa_result = self.qaoa_solver.solve(Q, C0, exact_cost=exact_cost)
        return qaoa_result["best_bitstring"], qaoa_result
