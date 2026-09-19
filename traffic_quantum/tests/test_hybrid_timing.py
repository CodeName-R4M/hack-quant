"""Test suite for Phase 5 Hybrid Timing (Quantum phase selection + Classical duration clamping)."""

import pytest
from traffic_quantum.config import DEFAULT_CONFIG
from traffic_quantum.controllers.hybrid import HybridController
from traffic_quantum.network import RoadNetwork
from traffic_quantum.simulator import TrafficSimulator


def test_classical_green_duration_clamping():
    """Verify that classical timing clamps green duration between min_green and max_green."""
    network = RoadNetwork()
    sim = TrafficSimulator(network=network, seed=42)
    controller = HybridController(network, solver_mode="brute_force")

    cfg = DEFAULT_CONFIG.hybrid

    # Case 1: Empty queue -> base_green clamped to [min_green, max_green]
    node = 0
    dur_empty = controller._compute_classical_duration(node, phase=0, simulator=sim)
    assert dur_empty == cfg.base_green_sec
    assert cfg.min_green_sec <= dur_empty <= cfg.max_green_sec

    # Case 2: Moderate queue -> base + k * queue
    # Add 10 cars to North approach
    for _ in range(10):
        sim.spawn_vehicle(origin_node=node, approach="N")
    dur_med = controller._compute_classical_duration(node, phase=0, simulator=sim)
    expected_med = int(round(cfg.base_green_sec + cfg.k_queue * 10))
    assert dur_med == expected_med
    assert cfg.min_green_sec < dur_med < cfg.max_green_sec

    # Case 3: Massive queue (e.g., 100 cars) -> should clamp to max_green
    for _ in range(90):
        sim.spawn_vehicle(origin_node=node, approach="N")
    dur_max = controller._compute_classical_duration(node, phase=0, simulator=sim)
    assert dur_max == cfg.max_green_sec


def test_hybrid_controller_qaoa_execution_in_sim():
    """Verify end-to-end execution of HybridController with QAOA backend in simulation loop."""
    network = RoadNetwork()
    sim = TrafficSimulator(network=network, seed=42)
    controller = HybridController(network, solver_mode="qaoa")

    for tick in range(35):
        phases = controller.get_phases(tick, sim)
        sim.step(phases)

    assert len(controller.optimization_history) >= 1
    last_opt = controller.optimization_history[-1]
    assert last_opt["solver"] == "qaoa"
    assert "approximation_ratio" in last_opt
    assert "probabilities" in last_opt


def test_reopt_interval_governs_phase_switching():
    """Verify that reopt_interval_sec controls simulation re-optimization frequency.
    
    Verifies that while classical clamp() computes theoretical green extension durations,
    actual phase updates in the simulation loop occur at every reopt_interval_sec.
    """
    network = RoadNetwork()
    sim = TrafficSimulator(network=network, seed=42)
    controller = HybridController(network, solver_mode="brute_force")

    # Reopt interval is 10s
    reopt_interval = controller.config.hybrid.reopt_interval_sec
    assert reopt_interval == 10

    # Step through simulation and record optimization ticks
    opt_ticks = []
    for tick in range(35):
        phases = controller.get_phases(tick, sim)
        sim.step(phases)
        if controller.optimization_history and controller.optimization_history[-1]["tick"] == tick:
            if not opt_ticks or opt_ticks[-1] != tick:
                opt_ticks.append(tick)

    # Optimization must trigger at tick 0, 10, 20, 30
    assert opt_ticks == [0, 10, 20, 30]

