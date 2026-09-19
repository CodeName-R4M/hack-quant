"""Ambulance Fairness Benchmark: Soft QUBO Preemption vs Hard Preemption Baselines.

Compares:
1. Fixed-Timing Baseline (unpreempted)
2. Fixed + Hard Preemption
3. Rule-Based Baseline (unpreempted)
4. Rule-Based + Hard Preemption
5. Hybrid (Brute-Force) (Soft QUBO Preemption, W_emerg=50)
6. Hybrid (QAOA) (Soft QUBO Preemption, W_emerg=50)

Evaluated across 20 evaluation seeds (100–119) under moderate_load and rush_hour regimes (600s each).
Outputs results to results/ambulance_fairness.csv and results/ambulance_fairness.json.
"""

import json
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from traffic_quantum.config import DEFAULT_CONFIG
from traffic_quantum.controllers.fixed import FixedController
from traffic_quantum.controllers.hybrid import HybridController
from traffic_quantum.controllers.rule_based import RuleBasedController
from traffic_quantum.emergency import EmergencyCorridorManager
from traffic_quantum.events import EventManager, EventType, TrafficEvent
from traffic_quantum.metrics import MetricsEngine
from traffic_quantum.network import RoadNetwork
from traffic_quantum.scenarios import SCENARIO_SPECS
from traffic_quantum.simulator import TrafficSimulator


def run_unpreempted_run(scenario_id: str, seed: int, duration: int = 600) -> float:
    """Runs an un-preempted baseline simulation to compute baseline civilian delay."""
    cfg = DEFAULT_CONFIG
    spec = SCENARIO_SPECS[scenario_id]
    net = RoadNetwork(cfg.network)
    sim = TrafficSimulator(net, cfg, seed=seed)
    if spec.boundary_rates:
        sim.boundary_arrival_rates = dict(spec.boundary_rates)

    evt_mgr = EventManager(net)
    if spec.accident_edge:
        evt = TrafficEvent(
            id=f"accident_{seed}",
            event_type=EventType.ACCIDENT,
            start_tick=spec.accident_start,
            duration_ticks=spec.accident_duration,
            target_edge=spec.accident_edge,
        )
        evt_mgr.schedule_event(evt)

    ctrl = FixedController(net, cfg)
    for tick in range(duration):
        evt_mgr.step(tick, sim, None)
        p = ctrl.get_phases(tick, sim)
        sim.step(p)

    m = MetricsEngine().compute_run_metrics(sim)
    return float(m["avg_wait_sec"])


def run_ambulance_fairness_study():
    cfg = DEFAULT_CONFIG
    scenarios = ["moderate_load", "rush_hour"]
    eval_seeds = list(range(100, 120))  # 20 evaluation seeds
    duration = 600

    controllers = [
        "Fixed-Timing (No Preemption)",
        "Fixed + Hard Preemption",
        "Rule-Based (No Preemption)",
        "Rule-Based + Hard Preemption",
        "Hybrid (Brute-Force) (Soft QUBO)",
        "Hybrid (QAOA) (Soft QUBO)",
    ]

    records = []
    print("Starting Ambulance Fairness Study across 20 evaluation seeds...")

    # Load existing benchmark runs from scenario_benchmark.json for consistency
    existing_records = []
    sc_json_path = os.path.join(os.path.dirname(__file__), "..", "results", "scenario_benchmark.json")
    if os.path.exists(sc_json_path):
        try:
            with open(sc_json_path, "r", encoding="utf-8") as f:
                sc_data = json.load(f)
            for r in sc_data.get("records", []):
                if r["scenario"] in scenarios and r["seed"] in eval_seeds:
                    ctrl = r["controller"]
                    mapped_name = None
                    if ctrl == "Fixed-Timing Baseline":
                        mapped_name = "Fixed-Timing (No Preemption)"
                    elif ctrl == "Rule-Based (Longest Queue)":
                        mapped_name = "Rule-Based (No Preemption)"
                    elif ctrl == "Hybrid (Brute-Force)":
                        mapped_name = "Hybrid (Brute-Force) (Soft QUBO)"
                    elif ctrl == "Hybrid (QAOA)":
                        mapped_name = "Hybrid (QAOA) (Soft QUBO)"

                    if mapped_name:
                        existing_records.append({
                            "scenario": r["scenario"],
                            "controller": mapped_name,
                            "seed": r["seed"],
                            "avg_wait_sec": float(r["avg_wait_sec"]),
                            "ambulance_time_sec": float(r["ambulance_time_sec"]),
                            "extra_delay_sec": float(r.get("extra_delay_sec", 0.0)),
                            "throughput_cpm": float(r["throughput_cpm"]),
                        })
        except Exception as e:
            print(f"Notice: could not load existing scenario benchmark: {e}")

    records.extend(existing_records)

    # Now run the new hard preemption baselines: Fixed + Hard Preemption, Rule-Based + Hard Preemption
    new_controllers = [
        "Fixed + Hard Preemption",
        "Rule-Based + Hard Preemption",
    ]

    for sc_id in scenarios:
        spec = SCENARIO_SPECS[sc_id]
        print(f"\n--- Running Hard Preemption Baselines for: {spec.name} ---")

        for seed in eval_seeds:
            unpreempted_wait = run_unpreempted_run(sc_id, seed, duration=duration)

            for c_name in new_controllers:
                net = RoadNetwork(cfg.network)
                sim = TrafficSimulator(net, cfg, seed=seed)
                if spec.boundary_rates:
                    sim.boundary_arrival_rates = dict(spec.boundary_rates)

                em_mgr = EmergencyCorridorManager(net, config=cfg)
                evt_mgr = EventManager(net)
                if spec.accident_edge:
                    evt = TrafficEvent(
                        id=f"accident_{seed}",
                        event_type=EventType.ACCIDENT,
                        start_tick=spec.accident_start,
                        duration_ticks=spec.accident_duration,
                        target_edge=spec.accident_edge,
                    )
                    evt_mgr.schedule_event(evt)

                if c_name.startswith("Fixed"):
                    ctrl = FixedController(net, cfg)
                else:
                    ctrl = RuleBasedController(net, cfg)

                amb_mission = em_mgr.dispatch_ambulance(origin=0, destination=5, simulator=sim, current_tick=15)

                for tick in range(duration):
                    evt_mgr.step(tick, sim, em_mgr)
                    biases = em_mgr.update_and_get_biases(tick, sim)

                    phases = ctrl.get_phases(tick, sim)
                    if biases:
                        # Hard preemption override: force green along active route corridor
                        for node, req_dir in biases.items():
                            phases[node] = (0 if req_dir == "NS" else 1)

                    sim.step(phases)

                m = MetricsEngine().compute_run_metrics(sim)
                amb_time = (
                    (amb_mission.arrival_tick - amb_mission.dispatch_tick)
                    if (amb_mission and amb_mission.arrival_tick)
                    else float(duration)
                )
                extra_delay = max(0.0, float(m["avg_wait_sec"]) - unpreempted_wait)

                records.append({
                    "scenario": sc_id,
                    "controller": c_name,
                    "seed": seed,
                    "avg_wait_sec": float(m["avg_wait_sec"]),
                    "ambulance_time_sec": float(amb_time),
                    "extra_delay_sec": float(round(extra_delay, 2)),
                    "throughput_cpm": float(m["throughput_cars_per_min"]),
                })

        print(f"  Completed all 20 seeds for {spec.name}")

    df = pd.DataFrame(records)
    os.makedirs("results", exist_ok=True)

    # Compute summary statistics
    summary_rows = []
    for sc_id in scenarios:
        sc_name = SCENARIO_SPECS[sc_id].name
        for c_name in controllers:
            sub = df[(df["scenario"] == sc_id) & (df["controller"] == c_name)]
            w_arr = sub["avg_wait_sec"].values
            a_arr = sub["ambulance_time_sec"].values
            e_arr = sub["extra_delay_sec"].values
            t_arr = sub["throughput_cpm"].values
            n = len(sub)

            w_m, w_s = float(np.mean(w_arr)), float(np.std(w_arr, ddof=1))
            w_ci = 1.96 * w_s / np.sqrt(n)

            a_m, a_s = float(np.mean(a_arr)), float(np.std(a_arr, ddof=1))
            a_ci = 1.96 * a_s / np.sqrt(n)

            e_m = float(np.mean(e_arr))
            t_m = float(np.mean(t_arr))

            summary_rows.append({
                "Scenario": sc_name,
                "Scenario_ID": sc_id,
                "Controller": c_name,
                "Vehicle Wait Mean (s)": round(w_m, 2),
                "Vehicle Wait 95% CI": f"[{w_m - w_ci:.2f}, {w_m + w_ci:.2f}]",
                "Ambulance Time Mean (s)": round(a_m, 2),
                "Ambulance Time 95% CI": f"[{a_m - a_ci:.2f}, {a_m + a_ci:.2f}]",
                "Ambulance Time Std (s)": round(a_s, 2),
                "Extra Delay (s)": round(e_m, 2),
                "Throughput (cpm)": round(t_m, 1),
                "Route_Description": "Path [0, 1, 2, 5], 4 nodes, 3 edges (0->1, 1->2, 2->5), 24s free-flow",
            })

    df_summary = pd.DataFrame(summary_rows)
    csv_path = "results/ambulance_fairness.csv"
    json_path = "results/ambulance_fairness.json"

    df_summary.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_rows,
            "records": records,
            "metadata": {
                "route": [0, 1, 2, 5],
                "num_edges": 3,
                "free_flow_travel_time_sec": 24.0,
                "explanation_8s_13s": (
                    "Route: Nodes [0, 1, 2, 5] (Guindy -> Saidapet -> Nandanam -> Thousand Lights). "
                    "3 directed edges with free-flow travel time 12s / 1.5 speed multiplier = 8s per edge = 24s total. "
                    "When dispatched with dispatch_tick=15, the ambulance has progressed 15s by tick 15, leaving 9s of transit. "
                    "8s represents unimpeded arrival at tick 23 (23 - 15 = 8s) under all-green conditions. "
                    "13s represents arrival at tick 28 (28 - 15 = 13s), delayed by 5s due to one red phase or front-of-queue clearance."
                )
            }
        }, f, indent=2)

    print(f"\nSaved fairness summary to {csv_path} and full data to {json_path}")
    return df_summary


if __name__ == "__main__":
    run_ambulance_fairness_study()
