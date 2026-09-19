"""Traffic simulation engine (pure Python tick-based simulation).

1 tick = 1 simulated second.
Guarantees 100% determinism given a random seed, allowing direct,
apples-to-apples performance comparisons across all signal control strategies.
"""

from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Set, Tuple

from traffic_quantum.config import DEFAULT_CONFIG, MasterConfig
from traffic_quantum.network import RoadNetwork


@dataclass
class Vehicle:
    """Represents an individual vehicle in the simulation."""
    id: int
    entry_tick: int
    origin: int
    target_destination: Optional[int] = None
    is_emergency: bool = False
    waiting_ticks: int = 0
    travel_ticks: int = 0
    current_intersection: int = 0
    current_approach: str = "N"  # 'N', 'S', 'E', 'W'
    in_transit_ticks_remaining: int = 0
    completed: bool = False
    exit_tick: Optional[int] = 0


class TrafficSimulator:
    """Tick-based traffic simulation engine for a multi-intersection grid."""

    def __init__(self, network: Optional[RoadNetwork] = None, config: Optional[MasterConfig] = None, seed: int = 42):
        self.config = config or DEFAULT_CONFIG
        self.network = network or RoadNetwork(self.config.network)
        self.seed = seed
        self.rng = random.Random(seed)
        
        self.current_tick: int = 0
        self.vehicle_id_counter: int = 0
        
        # Approach queues: {node_id: {'N': [Vehicle], 'S': [Vehicle], 'E': [Vehicle], 'W': [Vehicle]}}
        self.queues: Dict[int, Dict[str, List[Vehicle]]] = {
            node: {"N": [], "S": [], "E": [], "W": []}
            for node in self.network.graph.nodes
        }
        
        # In-transit vehicles on directed edges: {(u, v): [Vehicle]}
        self.in_transit: Dict[Tuple[int, int], List[Vehicle]] = {
            (u, v): [] for u, v in self.network.graph.edges
        }
        
        # Current signal phases: {node_id: 0 (NS Green) or 1 (EW Green)}
        self.signal_phases: Dict[int, int] = {
            node: self.config.simulation.phase_ns_green
            for node in self.network.graph.nodes
        }
        
        # Metrics and tracking
        self.completed_vehicles: List[Vehicle] = []
        self.throughput_history: List[int] = []  # Exits per tick
        self.queue_length_history: List[Dict[int, int]] = []  # Total queued cars per node per tick
        self.spillback_events_count: int = 0

        # Boundary entry points
        self.boundary_approaches = self.network.get_boundary_approaches()

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets the simulator to time 0 with clean state."""
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.current_tick = 0
        self.vehicle_id_counter = 0
        self.queues = {
            node: {"N": [], "S": [], "E": [], "W": []}
            for node in self.network.graph.nodes
        }
        self.in_transit = {
            (u, v): [] for u, v in self.network.graph.edges
        }
        self.signal_phases = {
            node: self.config.simulation.phase_ns_green
            for node in self.network.graph.nodes
        }
        self.completed_vehicles = []
        self.throughput_history = []
        self.queue_length_history = []
        self.spillback_events_count = 0

    def set_signal_phases(self, phases: Dict[int, int]) -> None:
        """Updates intersection signal phases. 0 = NS green, 1 = EW green."""
        for node_id, phase in phases.items():
            if node_id in self.signal_phases:
                self.signal_phases[node_id] = phase

    def spawn_vehicle(
        self,
        origin_node: int,
        approach: str,
        destination: Optional[int] = None,
        is_emergency: bool = False
    ) -> Optional[Vehicle]:
        """Spawns a new vehicle directly into an approach queue."""
        self.vehicle_id_counter += 1
        veh = Vehicle(
            id=self.vehicle_id_counter,
            entry_tick=self.current_tick,
            origin=origin_node,
            target_destination=destination,
            is_emergency=is_emergency,
            current_intersection=origin_node,
            current_approach=approach,
        )
        self.queues[origin_node][approach].append(veh)
        return veh

    def _generate_boundary_arrivals(self) -> None:
        """Stochastically spawns vehicles at network perimeter boundaries based on arrival rate."""
        rate = self.config.simulation.base_arrival_rate
        for node_id, approach in self.boundary_approaches:
            if self.rng.random() < rate:
                self.spawn_vehicle(node_id, approach)

    def _get_target_neighbor(self, node: int, approach: str) -> Optional[Tuple[int, str]]:
        """Determines the downstream intersection and target approach queue when moving through a green light.
        
        Movement mapping:
        - Approaching from North ('N'): moving South -> target neighbor (node + cols) if exists
        - Approaching from South ('S'): moving North -> target neighbor (node - cols) if exists
        - Approaching from East ('E'): moving West -> target neighbor (node - 1) if exists in same row
        - Approaching from West ('W'): moving East -> target neighbor (node + 1) if exists in same row
        """
        for nbr in self.network.graph.neighbors(node):
            edge_data = self.network.graph[node][nbr]
            # If car enters from 'N', it is moving along direction 'S'
            if approach == "N" and edge_data["direction"] == "S":
                return nbr, edge_data["target_approach"]
            elif approach == "S" and edge_data["direction"] == "N":
                return nbr, edge_data["target_approach"]
            elif approach == "E" and edge_data["direction"] == "W":
                return nbr, edge_data["target_approach"]
            elif approach == "W" and edge_data["direction"] == "E":
                return nbr, edge_data["target_approach"]
        return None

    def _discharge_queues(self) -> int:
        """Discharges vehicles from green approaches.
        
        Discharges ~1 vehicle every 2 seconds (tick % 2 == 0).
        Enforces downstream edge capacity and spillback blocking.
        """
        discharged_count = 0
        should_discharge = (self.current_tick % self.config.simulation.discharge_interval_ticks == 0)

        for node_id in self.network.graph.nodes:
            phase = self.signal_phases[node_id]
            # Active green approaches for this intersection
            if phase == self.config.simulation.phase_ns_green:
                green_approaches = ["N", "S"]
            else:
                green_approaches = ["E", "W"]

            for app in green_approaches:
                queue = self.queues[node_id][app]
                if not queue or not should_discharge:
                    continue

                downstream = self._get_target_neighbor(node_id, app)
                if downstream is None:
                    # Car departs the network boundaries (Throughput)
                    veh = queue.pop(0)
                    veh.completed = True
                    veh.exit_tick = self.current_tick
                    veh.travel_ticks = self.current_tick - veh.entry_tick
                    self.completed_vehicles.append(veh)
                    discharged_count += 1
                else:
                    target_node, target_app = downstream
                    edge_data = self.network.graph[node_id][target_node]
                    cap = edge_data["capacity"]
                    
                    # Current load on downstream link = vehicles in-transit + target approach queue
                    current_load = (
                        len(self.in_transit[(node_id, target_node)]) +
                        len(self.queues[target_node][target_app])
                    )
                    
                    if current_load < cap and edge_data["status"] != "closed":
                        # Road segment has space: discharge vehicle into transit
                        veh = queue.pop(0)
                        veh.current_intersection = target_node
                        veh.current_approach = target_app
                        veh.in_transit_ticks_remaining = max(1, edge_data["free_flow_travel_time"])
                        self.in_transit[(node_id, target_node)].append(veh)
                        discharged_count += 1
                    else:
                        # Spillback: downstream road is saturated or closed!
                        self.spillback_events_count += 1

        return discharged_count

    def _advance_in_transit(self) -> None:
        """Advances vehicles traversing road segments and transfers them to target queues."""
        for (u, v), vehicles in self.in_transit.items():
            remaining_transit = []
            for veh in vehicles:
                veh.in_transit_ticks_remaining -= 1
                veh.travel_ticks += 1
                if veh.in_transit_ticks_remaining <= 0:
                    # Vehicle arrives at target intersection queue
                    self.queues[v][veh.current_approach].append(veh)
                else:
                    remaining_transit.append(veh)
            self.in_transit[(u, v)] = remaining_transit

    def _accumulate_waiting_times(self) -> None:
        """Adds waiting ticks for all vehicles currently queued at intersections."""
        for node_id, approaches in self.queues.items():
            for app, veh_list in approaches.items():
                for veh in veh_list:
                    veh.waiting_ticks += 1
                    veh.travel_ticks += 1

    def step(self, signal_phases: Optional[Dict[int, int]] = None) -> Dict[str, object]:
        """Advances the simulation by 1 second (1 tick).
        
        Args:
            signal_phases: Optional dictionary of {node_id: phase (0 or 1)}
            
        Returns:
            Dictionary with current tick summary metrics.
        """
        if signal_phases:
            self.set_signal_phases(signal_phases)

        self._generate_boundary_arrivals()
        discharged = self._discharge_queues()
        self._advance_in_transit()
        self._accumulate_waiting_times()

        # Record metrics for current tick
        self.throughput_history.append(discharged)
        node_queues = {
            node: sum(len(q) for q in self.queues[node].values())
            for node in self.network.graph.nodes
        }
        self.queue_length_history.append(node_queues)
        self.current_tick += 1

        return {
            "tick": self.current_tick,
            "discharged": discharged,
            "total_queued": sum(node_queues.values()),
            "completed_count": len(self.completed_vehicles),
        }

    def get_total_queue(self, node_id: int) -> int:
        """Returns the total number of queued vehicles at a given intersection."""
        return sum(len(q) for q in self.queues[node_id].values())

    def get_approach_queue_lengths(self, node_id: int) -> Dict[str, int]:
        """Returns the breakdown of queue lengths for N, S, E, W at an intersection."""
        return {app: len(q) for app, q in self.queues[node_id].items()}

    def get_average_wait_time(self) -> float:
        """Calculates average waiting time across all vehicles currently in the system and completed."""
        all_waits = [v.waiting_ticks for v in self.completed_vehicles]
        for node_queues in self.queues.values():
            for q in node_queues.values():
                all_waits.extend(v.waiting_ticks for v in q)
        return (sum(all_waits) / len(all_waits)) if all_waits else 0.0
