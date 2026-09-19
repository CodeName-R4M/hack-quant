"""Global Configuration for Quantum-Enhanced Adaptive Urban Traffic Optimization.

All tunable parameters, simulation constants, QUBO weights, QAOA hyper-parameters,
metrics coefficients, and security policies are centralized here.
No magic numbers are scattered across modules.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class NetworkConfig:
    """Road network graph topology and capacity parameters."""
    grid_rows: int = 2
    grid_cols: int = 3
    num_intersections: int = 6  # grid_rows * grid_cols (supports 4 to 8)
    default_road_length_m: float = 150.0
    default_capacity: int = 20  # Max vehicles queued/in-flight on an edge
    free_flow_speed_mps: float = 12.5  # ~45 km/h
    free_flow_travel_time_sec: int = 12  # ticks required to traverse road segment

    # Chennai real junction mapping (lat, lon) for 2x3 grid
    chennai_junctions: Dict[int, Dict[str, object]] = field(default_factory=lambda: {
        0: {"name": "Guindy Kathipara", "lat": 13.0067, "lon": 80.2030},
        1: {"name": "Saidapet Metro", "lat": 13.0185, "lon": 80.2225},
        2: {"name": "Nandanam Signal", "lat": 13.0310, "lon": 80.2370},
        3: {"name": "T. Nagar Panagal Park", "lat": 13.0405, "lon": 80.2337},
        4: {"name": "Anna Salai DMS", "lat": 13.0450, "lon": 80.2480},
        5: {"name": "Thousand Lights", "lat": 13.0560, "lon": 80.2540},
    })


@dataclass
class SimulationConfig:
    """Tick-based traffic simulation parameters."""
    tick_duration_sec: float = 1.0  # 1 tick = 1 second
    default_seed: int = 42
    default_sim_duration_sec: int = 600  # 10 minutes benchmark run
    
    # Inflow arrival rates (Poisson-like probability per tick at boundary entry queues)
    base_arrival_rate: float = 0.35  # ~21 cars/minute per entry point
    
    # Discharge characteristics
    discharge_interval_ticks: int = 2  # Discharges ~1 vehicle per 2 seconds (0.5 car/s)
    
    # Approaches per junction: 0=North, 1=South, 2=East, 3=West
    # Phases: 0 = North-South GREEN (EW RED), 1 = East-West GREEN (NS RED)
    phase_ns_green: int = 0
    phase_ew_green: int = 1


@dataclass
class BaselineConfig:
    """Parameters for classical baseline controllers."""
    # Fixed controller: 30s NS, 30s EW
    fixed_ns_green_sec: int = 30
    fixed_ew_green_sec: int = 30
    fixed_cycle_sec: int = 60

    # Rule-based controller: evaluates every 10s, min green 10s to prevent flicker
    rule_eval_interval_sec: int = 10
    rule_min_green_sec: int = 10


@dataclass
class QUBOConfig:
    """Weights and penalty coefficients for QUBO cost Hamiltonian."""
    w_queue: float = 1.0          # Linear penalty for queued cars facing red
    w_coord: float = 0.2          # Quadratic coupling (tuned on training seeds 1-5)
    w_spillback: float = 0.5      # Downstream spillback penalty (tuned on training seeds 1-5)
    w_emergency: float = 50.0     # Heavy linear bias when emergency vehicle approaches
    w_pedestrian: float = 2.0     # Linear penalty for queued pedestrian waiting
    w_switch: float = 2.5         # Switching penalty (cost for flipping signal phase)
    spillback_threshold: float = 0.80  # Queue / capacity fraction considered spillback hazard


@dataclass
class QAOAConfig:
    """PennyLane QAOA circuit hyperparameters and optimizer settings."""
    p_layers: int = 2             # QAOA layers (p = 2 or 3)
    max_iterations: int = 35      # Classical optimizer steps (COBYLA)
    step_size: float = 0.1        # Initial step size for optimization
    top_k_bitstrings: int = 4     # Inspect top-k most probable bitstrings
    warm_start: bool = True       # Cache optimal angles gamma/beta across rounds
    device_name: str = "default.qubit"
    shots: int = 1000             # For measurement sampling


@dataclass
class HybridTimingConfig:
    """Classical timing engine for dynamic phase durations."""
    reopt_interval_sec: int = 20  # Re-optimize phases every 20-30 seconds (stabilizes phase switching)
    base_green_sec: int = 15      # Baseline green time
    k_queue: float = 0.5          # Extension coefficient per queued vehicle
    min_green_sec: int = 10       # Minimum green duration
    max_green_sec: int = 45       # Maximum green duration


@dataclass
class EmergencyConfig:
    """Emergency Green Corridor routing and preemption parameters."""
    eta_threshold_sec: float = 45.0  # Apply preemption when ambulance ETA <= 45s
    speed_multiplier: float = 1.5    # Ambulances travel faster under green corridor
    max_preemption_duration_sec: int = 90  # Hard cutoff to prevent endless starvation
    default_origin: int = 0
    default_destination: int = 5
    hard_preemption_mode: bool = False
    max_concurrent_emergencies: int = 2


@dataclass
class PedestrianConfig:
    """Pedestrian demand and crossing parameters."""
    arrival_rate: float = 0.05       # Probability per second of a pedestrian arriving at an intersection
    max_wait_sec: int = 45          # Maximum acceptable wait before forced walk phase
    walk_duration_sec: int = 10     # Duration of forced walk phase
    w_pedestrian: float = 2.0       # Weight for pedestrian wait penalty in optimization


@dataclass
class MetricsConfig:
    """Environmental and performance evaluation constants."""
    idle_fuel_rate_l_per_hr: float = 0.8  # Idle consumption: 0.8 L/hour
    co2_kg_per_l_petrol: float = 2.31      # 2.31 kg CO2 per litre petrol consumed


@dataclass
class SecurityConfig:
    """Cryptographic authorization and rate limiting."""
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET", "quant-traffic-jwt-key-chennai-sec-2026"))
    jwt_algorithm: str = "HS256"
    token_validity_sec: int = 600
    rate_limit_requests: int = 5
    rate_limit_window_sec: int = 60
    audit_log_file: str = "audit_log.jsonl"


@dataclass
class UIConfig:
    """Dashboard UI and tile server settings."""
    map_tile_provider: str = field(default_factory=lambda: os.getenv("MAP_TILE_PROVIDER", "carto_dark"))
    cartodb_api_key: str = field(default_factory=lambda: os.getenv("CARTODB_API_KEY", ""))
    google_maps_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_MAPS_API_KEY", ""))


@dataclass
class MasterConfig:
    """Master configuration tree."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    qubo: QUBOConfig = field(default_factory=QUBOConfig)
    qaoa: QAOAConfig = field(default_factory=QAOAConfig)
    hybrid: HybridTimingConfig = field(default_factory=HybridTimingConfig)
    emergency: EmergencyConfig = field(default_factory=EmergencyConfig)
    pedestrian: PedestrianConfig = field(default_factory=PedestrianConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    ui: UIConfig = field(default_factory=UIConfig)


# Global default instance
DEFAULT_CONFIG = MasterConfig()
