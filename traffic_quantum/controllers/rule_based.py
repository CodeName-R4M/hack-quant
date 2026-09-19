"""Rule-Based (Longest-Queue-First) Baseline Controller.

Evaluates approach queues every 10 seconds and grants green to the direction
with the larger queue, while enforcing a minimum green time to avoid signal flicker.
"""

from typing import Dict, Optional
from traffic_quantum.config import DEFAULT_CONFIG, MasterConfig
from traffic_quantum.controllers.base import BaseController
from traffic_quantum.network import RoadNetwork


class RuleBasedController(BaseController):
    """Adaptive heuristic controller prioritizing the direction with the longest queue."""

    def __init__(self, network: RoadNetwork, config: Optional[MasterConfig] = None):
        super().__init__(network, config)
        self.name = "Rule-Based (Longest Queue)"
        self.eval_interval = self.config.baseline.rule_eval_interval_sec
        self.min_green = self.config.baseline.rule_min_green_sec
        
        # State tracking per intersection
        self.current_phases: Dict[int, int] = {
            node: self.config.simulation.phase_ns_green
            for node in self.network.graph.nodes
        }
        self.last_switch_tick: Dict[int, int] = {
            node: 0 for node in self.network.graph.nodes
        }

    def reset(self) -> None:
        """Resets controller state."""
        self.current_phases = {
            node: self.config.simulation.phase_ns_green
            for node in self.network.graph.nodes
        }
        self.last_switch_tick = {
            node: 0 for node in self.network.graph.nodes
        }

    def get_phases(self, current_tick: int, simulator) -> Dict[int, int]:
        """Periodically evaluates queue lengths and updates phases if minimum green duration has elapsed."""
        if current_tick % self.eval_interval == 0:
            for node in self.network.graph.nodes:
                q_lengths = simulator.get_approach_queue_lengths(node)
                q_ns = q_lengths.get("N", 0) + q_lengths.get("S", 0)
                q_ew = q_lengths.get("E", 0) + q_lengths.get("W", 0)

                desired_phase = self.current_phases[node]
                if q_ns > q_ew:
                    desired_phase = self.config.simulation.phase_ns_green
                elif q_ew > q_ns:
                    desired_phase = self.config.simulation.phase_ew_green

                time_since_last_switch = current_tick - self.last_switch_tick[node]
                if desired_phase != self.current_phases[node] and time_since_last_switch >= self.min_green:
                    self.current_phases[node] = desired_phase
                    self.last_switch_tick[node] = current_tick

        return dict(self.current_phases)
