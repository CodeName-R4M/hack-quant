# Quantum Traffic Brain

> **Quantum-Enhanced Adaptive Urban Traffic Optimization System**
> Built for Hack-Quant · Real Chennai junctions · YOLOv8 + QAOA

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Project Structure](#project-structure)
4. [Module Reference](#module-reference)
5. [Dashboard Pages](#dashboard-pages)
6. [End-to-End System Flow](#end-to-end-system-flow)
7. [Quantum Optimization Pipeline](#quantum-optimization-pipeline)
8. [Emergency Corridor System](#emergency-corridor-system)
9. [Vehicle Detection Pipeline](#vehicle-detection-pipeline)
10. [Configuration Parameters](#configuration-parameters)
11. [Installation & Running](#installation--running)
12. [Known Limitations](#known-limitations)

---

## Overview

**Quantum Traffic Brain** is a real-time, browser-based traffic signal optimization dashboard.
It simulates a 2x3 grid of six real Chennai intersections, optimizes signal phases using
**QAOA** (Quantum Approximate Optimization Algorithm) encoded as a **QUBO** problem,
handles emergency ambulance routing with dynamic signal preemption, and accepts live vehicle
counts from CCTV images/videos using **YOLOv8n** computer vision.

The entire application runs in a single **Streamlit** process — no database needed.

### Where Quantum is Used and Why We Don't Claim Advantage
In this project, quantum computing via QAOA (Quantum Approximate Optimization Algorithm) is utilized specifically to solve the quadratic unconstrained binary optimization (QUBO) problem representing coordinated traffic phase selection across the network. Because classical brute-force can trivially solve a 6-qubit (64-state) system in less than a millisecond on a laptop, **we make no claim of quantum advantage or supremacy**. Instead, this system serves as a **quantum-ready pipeline validated on a simulator**: it formulates the combinatorial traffic problem natively for quantum processors so that as physical fault-tolerant quantum hardware scales, the identical mathematical pipeline can target actual quantum processing units (QPUs) for large-scale city grids where classical combinatorial search becomes intractable ($O(2^N)$ wall-clock explosion).

---

## Architecture Diagram

```
+---------------------------------------------------------------+
|                     STREAMLIT DASHBOARD                       |
|  +---------------+  +---------------------+  +------------+  |
|  |   Sidebar     |  |  Live Canvas Tab    |  | Diagnostics|  |
|  |  (Controls)   |  |  (HTML/JS iframe)   |  | (Plotly)   |  |
|  +-------+-------+  +---------+-----------+  +------+-----+  |
|          |                    |                      |        |
|  +-------+--------------------+----------------------+------+ |
|  |                    Python Backend                         | |
|  |  RoadNetwork <-> TrafficSimulator <-> HybridController   | |
|  |      |                  |                    |            | |
|  |  EmergencyMgr <-> EventsEngine <-> QUBO + QAOA/BF       | |
|  |      |                  |                                 | |
|  |  MetricsEngine     SecurityService     VehicleDetector   | |
|  +-------------------------------------------------------+--+ |
+---------------------------------------------------------------+
```

---

## Project Structure

```
hack-quant/
├── dashboard.py              <- Main Streamlit entrypoint (Simulation + Hospital Google Maps routing)
├── benchmark.py              <- CLI multi-seed benchmark runner
├── yolov8n.pt                <- YOLOv8 nano weights (6.2 MB, auto-downloaded)
├── results/                  <- Empirical benchmarks & scientific artifacts (JSON, PNG, CSV)
└── traffic_quantum/
    ├── config.py             <- All constants, weights, and hyperparameters
    ├── network.py            <- NetworkX road graph (2x3 directed grid, Chennai coordinates)
    ├── simulator.py          <- Tick-based traffic simulation engine (queues & physics)
    ├── video_canvas.py       <- HTML/CSS/JS animated live canvas generator
    ├── vision.py             <- YOLOv8 + classical CV vehicle detector
    ├── emergency.py          <- Ambulance mission + green corridor manager + conflict arbiter
    ├── hospital_maps.py      <- Standalone Paramedic Hospital Google Maps console (/?page=hospital_maps)
    ├── signal_interface.py   <- Hardware-ready abstraction, Conflict Monitor, Watchdog
    ├── events.py             <- Dynamic events engine (accidents, surges, closures)
    ├── metrics.py            <- Performance + environmental metrics engine
    ├── security.py           <- JWT auth, rate limiting, append-only SHA-256 audit logging
    ├── canvas.py             <- Plotly static diagnostic canvas builder
    ├── benchmark.py          <- Multi-seed controller comparison runner
    ├── requirements.txt
    ├── controllers/
    │   ├── base.py           <- Abstract controller interface
    │   ├── hybrid.py         <- Quantum-classical hybrid controller (QAOA / Brute-Force)
    │   ├── rule_based.py     <- Greedy queue-length rule controller
    │   └── fixed.py          <- Fixed 30s/30s cycle baseline
    ├── quantum/
    │   ├── qubo.py           <- QUBO matrix builder from live traffic state
    │   ├── qaoa.py           <- Manual PennyLane QAOA solver (p=1 to 4)
    │   ├── ising.py          <- QUBO to Ising model converter
    │   ├── brute_force.py    <- Exact 2^N brute-force solver (verification)
    │   └── simulated_annealing.py <- Classical simulated annealing benchmark solver
    └── tests/                <- 33 unit tests across 13 test suites (100% passing)
        ├── test_baselines.py
        ├── test_emergency_conflict.py
        ├── test_emergency_corridor.py
        ├── test_events.py
        ├── test_hybrid_timing.py
        ├── test_pedestrians.py
        ├── test_qaoa_vs_brute_force.py
        ├── test_qubo_ising_equivalence.py
        ├── test_security.py
        ├── test_signal_interface.py
        ├── test_simulated_annealing.py
        ├── test_simulator_determinism.py
        └── test_vision.py
```

---

## Module Reference

### dashboard.py (Root Level)

**Role:** Single Streamlit entrypoint — all UI, state, buttons, and callbacks.

**Responsibilities:**
- Initialises `RoadNetwork`, `TrafficSimulator`, `HybridController`, `EmergencyCorridorManager`, `EventsEngine`, `MetricsEngine`, `SecurityService` in `st.session_state`
- Renders sidebar: city theme, traffic mode, live control, controller choice
- Builds main tabs: *Live Traffic Canvas* and *Diagnostic Analytics*
- Manages the *Traffic Inflow Expander* (gate inputs + CCTV upload + YOLOv8 detection)
- Handles ambulance dispatch, road events, metrics display, security panel, benchmark launcher
- Calls `sim.step()` + `controller.compute_phases(sim)` + `sim.set_signal_phases(phases)` each tick
- Detects Auto→Manual mode transition and calls `sim.clear_all_vehicles()` to wipe the map
- Injects AI-detected vehicle counts into simulator entry gates via `sim.inject_by_entry_name()`

---

### config.py

**Role:** Single source of truth for every constant and hyperparameter.

| Dataclass | What it configures |
|---|---|
| `NetworkConfig` | Grid size (2x3), road length (150 m), capacity (20 veh/edge), Chennai GPS coordinates |
| `SimulationConfig` | Tick duration (1 s), Poisson arrival rate (0.35/tick), discharge interval (2 ticks) |
| `QUBOConfig` | Cost weights: queue (1.0), coordination (2.5), spillback (6.0), emergency (50.0) |
| `QAOAConfig` | p=2 layers, 35 COBYLA iterations, 1000 shots, warm-start caching |
| `HybridTimingConfig` | Re-optimize every 30 s, base green 15 s, k=0.8 extension per queued vehicle |
| `EmergencyConfig` | ETA preemption threshold 45 s, ambulance speed x1.5, max preemption 90 s |
| `MetricsConfig` | Idle fuel 0.8 L/hr, CO2 2.31 kg/L petrol |
| `SecurityConfig` | HS256 JWT, 600 s token validity, 5 requests/60 s rate limit |

---

### network.py

**Role:** Builds and manages the directed road graph.

**Class:** `RoadNetwork`

- Creates a `networkx.DiGraph` with 6 nodes (intersections A-F)
- Maps each node to a real Chennai junction with GPS coordinates:
  - A: Guindy Kathipara | B: Saidapet Metro | C: Nandanam Signal
  - D: T. Nagar Panagal Park | E: Anna Salai DMS | F: Thousand Lights
- Adds bidirectional directed edges between all adjacent grid pairs (E-W and N-S)
- Each edge carries: `direction`, `target_approach`, `length_m` (150 m), `capacity` (20), `status`, `free_flow_travel_time` (12 ticks)
- `set_edge_status(u, v, status)` — halves capacity on `reduced`, zeros it on `closed`
- `get_boundary_approaches()` — returns all 10 perimeter gate (node, approach) pairs

Grid layout:
```
  W1->[A]<->[B]<->[C]<-E1
       |         |
  W2->[D]<->[E]<->[F]<-E2
  N1  N2  N3   (North entry)
  S1  S2  S3   (South entry)
```

---

### simulator.py

**Role:** Tick-based discrete-event traffic simulation engine.

**Data structures:**
- `queues[node_id][approach]` — list of `Vehicle` objects waiting at each junction approach (N/S/E/W)
- `in_transit[(u,v)]` — list of vehicles currently traveling between junctions
- `signal_phases[node_id]` — current green phase: 0=NS green, 1=EW green
- `completed_vehicles` — vehicles that have exited the network

**Key methods:**

| Method | What it does |
|---|---|
| `step()` | 1 tick: generate arrivals -> discharge queues -> advance in-transit |
| `spawn_vehicle(node, approach)` | Creates a vehicle in the specified approach queue |
| `spawn_at_entry_portal(portal_id, count)` | Spawns N vehicles at a named gate (N1-N3, S1-S3, W1-W2, E1-E2) |
| `inject_by_entry_name(name, count)` | Convenience alias for `spawn_at_entry_portal` |
| `clear_all_vehicles()` | Wipes all queues and in-transit lists without resetting signals or tick |
| `set_signal_phases(phases)` | Updates all intersection signal phases from controller output |
| `reset()` | Full state reset to tick 0 with clean queues |

**Simulation loop internals:**
1. `_generate_boundary_arrivals()` — 35% chance per boundary gate each tick (skipped in manual mode)
2. `_discharge_queues()` — every 2 ticks, for each green approach, move 1 vehicle to in_transit buffer
3. `_advance_in_transit()` — decrement travel ticks; when 0, move to next junction's queue or mark completed

---

### video_canvas.py

**Role:** Generates the self-contained HTML/CSS/JavaScript animated traffic canvas (1100x560 px).

**Function:** `generate_video_canvas_html(network, simulator, emergency_mgr, ...)`

**What it renders:**
- Dark `#0b0f19` background with neon-styled roads connecting junction nodes labeled A-F
- Traffic signals on the LEFT side of each junction as 3-lens vertical heads (red/amber/green) with animated glow
- Vehicles as colored dots flowing along Bezier curves between junctions, queuing at red signals, exiting the perimeter
- Ambulance rendered separately with flashing siren and highlighted green-corridor path
- Accidents shown as red X markers on blocked road segments
- HUD overlay: tick counter, vehicle count, active controller name

**How JS animation works:**
1. Python serialises entire sim state to `state_json` (nodes, edges, phases, vehicles, ambulance, autoFlow)
2. HTML blob injected into Streamlit `components.html()` iframe
3. Inside iframe, `requestAnimationFrame` drives animation at ~60 fps
4. `flowVehicles[]` holds all cars, each with: `x, y, heading, progress, edgeKey, gateId, isExiting, isGate`
5. Each frame: move vehicles along Bezier path, check signal, queue at red, route through junction, bias towards exits
6. If `autoFlow=true`, new vehicles auto-spawn at random gates every 90 frames
7. If `autoFlow=false` (manual mode), no auto-spawning; only Python-injected vehicles appear

**Bezier routing system:**
- `getInflowCurve(gateId)` — entry path from perimeter gate to junction stopline
- `getJunctionCurve(from, to)` — curved path through junction (supports left/right turns)
- `getOutflowCurve(nodeId, direction)` — exit path towards perimeter (70% exit bias to prevent center clustering)

---

### vision.py — YOLOv8 + Classical CV

**Role:** Detects and counts vehicles in uploaded images or videos.

**Primary engine: YOLOv8n (ultralytics)**
- Pre-trained on COCO, zero additional training needed
- Weights: `yolov8n.pt` (6.2 MB), auto-cached in `~/.cache/ultralytics/`
- COCO vehicle classes: `2=car`, `3=motorcycle`, `5=bus`, `7=truck`
- Confidence threshold: 0.30 | NMS IoU: 0.45
- Colour-coded boxes: Yellow=Car, Cyan=Motorcycle, Blue=Bus, Green=Truck

**Fallback engine: Classical CV (11 stages)**
Resize -> Denoise -> CLAHE -> Road surface mask -> Canny -> Fuse with mask -> Morph close ->
Dilate -> Stripe-killer -> Contour filter (solidity/AR/size) -> NMS

**Class `VehicleDetector`:**
- `decode_media_bytes(bytes, filename)` — decodes image or video (samples frame at 25% position)
- `detect_vehicles(img_bgr)` -> returns `(annotated_rgb, count, detections_list)`
- HUD shows engine tag: `[YOLOv8n]` or `[Classical CV]`

**`map_detected_count_to_entries(n, target, available)`:**
Distributes N detected vehicles across entry gates — all to one gate or round-robin.

---

### emergency.py

**Role:** Ambulance mission management and dynamic signal preemption.

**`AmbulanceMission` dataclass:**
`origin`, `destination`, `path` (node ID list), `driver_name`, `current_index`, `current_edge_progress_sec`, `completed`

`get_current_coordinates(network)` — interpolates real-time GPS position along route

**`EmergencyCorridorManager` methods:**

| Method | What it does |
|---|---|
| `dispatch(origin, destination, driver_name)` | Dijkstra to find shortest path, creates AmbulanceMission |
| `advance_mission(simulator)` | Moves ambulance progress along current edge each tick |
| `get_emergency_biases(simulator)` | Returns `{node_id: 'NS'/'EW'}` for nodes where ambulance ETA < 45 s |
| `restore_normal_signals(simulator, phases)` | Resets preempted signals once ambulance clears |

Ambulance destination is always the correct hospital junction, not a random node.

---

### hospital_maps.py

**Role:** Standalone Tactical EMS Paramedic Cockpit (`/?page=hospital_maps`), accessible via the new-tab button in Tab 2.

**Responsibilities:**
- Renders a full-page, self-contained ambulance in-cabin navigation interface
- **Real Google Maps Roadmap & Satellite**: Leaflet integration with Google Maps tiles (`https://mt1.google.com/vt/lyrs=m...`) and hybrid satellite layer
- **Real Address Search**: Search any Chennai location with OpenStreetMap Nominatim or click quick landmark chips (Marina Beach, T. Nagar, Central Station, Guindy Kathipara, Airport)
- **HTML5 GPS Geolocation**: "Use My GPS Location" button captures live physical coordinates via `navigator.geolocation`
- **Interactive Map Pinning**: Click anywhere directly on the Google Map to place or reposition ambulance pickup or hospital destination pins
- **Chennai Trauma Hospital Selector**: Dropdown covering Apollo Hospitals Greams Road, Rajiv Gandhi Govt General Hospital, Omandurar Multi Super Speciality, Kilpauk Medical College, MIOT International, Fortis Malar, Kauvery, or custom search
- **Real Road Network Driving Paths**: Queries OSRM driving engine (`https://router.project-osrm.org/route/v1/driving/...`) for exact street turning geometry, distance (km), and driving time
- **In-Transit Navigation Simulation**: "Start Emergency Journey" animates the ambulance along the real road polyline with dynamic speedometer (55–68 km/h), transit progress percentage, and turn-by-turn guidance

---

### events.py

**Role:** Real-time dynamic traffic disruption engine.

**Event types:**
- `EMERGENCY` — triggers ambulance dispatch
- `ROAD_CLOSURE` — sets edge capacity to 0
- `ACCIDENT` — halves edge capacity for a duration, auto-restores
- `CONGESTION_SURGE` — injects burst of vehicles at a boundary gate (3x rate)

**`EventsEngine` methods:**
- `trigger_event(event)` — activates event immediately
- `tick(simulator)` — manages event lifecycle (auto-resolve after duration)
- `resolve_event(event_id)` — restores road to normal, logs resolution

---

### metrics.py

**Role:** Computes performance and environmental metrics.

**`MetricsEngine.compute_run_metrics(simulator)` returns:**

| Metric | Unit | Description |
|---|---|---|
| `avg_wait_sec` | seconds | Mean vehicle waiting time |
| `max_wait_sec` | seconds | Worst-case waiting time |
| `throughput_per_min` | veh/min | Completed vehicles per minute |
| `avg_queue_len` | vehicles | Average junction queue depth |
| `idle_fuel_l` | litres | Estimated fuel wasted idling |
| `co2_kg` | kg | CO2 from idle fuel x 2.31 kg/L |
| `improvement_vs_fixed_pct` | % | Throughput gain vs fixed-timing baseline |

---

### security.py

**Role:** Protects emergency preemption from spoofed signals or DOS attacks.

| Feature | Detail |
|---|---|
| JWT authentication | HS256 tokens, 600 s validity |
| Rate limiting | Sliding window: max 5 requests per 60 s per client |
| Preemption cap | Hard limit of 90 s |
| Audit logging | Append-only JSONL with SHA-256 hash chain |
| Input validation | Validates all event payload fields |

---

### signal_interface.py

**Role:** Hardware-ready abstraction layer and physical safety monitors (Phase G).

**Components:**
- `SignalControllerInterface`: Vendor-agnostic abstract base class for traffic signal actuators (`set_phase`, `get_state`, `emergency_preempt`, `safe_fallback`). Enables NTCIP 1202 / NEMA TS2 integration without touching the optimization core.
- `SoftwareConflictMonitor`: Enforces mutually exclusive green phases (prevents concurrent N-S and E-W green lights), a **10-second minimum green constraint**, and a **2-second yellow clearance interval**.
- `HardwareWatchdog`: Heartbeat monitor tripping `safe_fallback()` into a fail-safe fixed-timing cycle if an optimization step times out (>15s).

---

### canvas.py

**Role:** Builds the static Plotly diagnostic canvas (Tab 2).

**Function:** `build_simulation_canvas(network, simulator, emergency_mgr, ...)`

Renders:
- Road segments as lines (green=open, red=closed, orange=accident)
- Junction nodes sized by queue length, labelled with name + live counts
- Signal state per node (green/red circle)
- Ambulance marker + hover tooltips
- Click-to-select junctions and roads for interactive event targeting

---

### controllers/

**`base.py` — `BaseController`**
Abstract interface. All controllers implement `compute_phases(simulator) -> Dict[int, int]`.

**`hybrid.py` — `HybridController`**
Primary quantum-classical controller. Every 30 s:
1. Builds QUBO matrix Q from live queue state
2. QAOA or brute-force solver -> optimal bitstring x*
3. Converts to phase map: x_i=0 -> NS green, x_i=1 -> EW green
4. Adaptive green duration: `clamp(15 + 0.8 * queue_count, 10, 45)` seconds

**`rule_based.py` — `RuleBasedController`**
Greedy: every 10 ticks, compares NS vs EW queue totals per junction, assigns green to heavier queue.

**`fixed.py` — `FixedTimingController`**
Baseline: alternates NS/EW green every 30 s on a fixed 60-second cycle.

---

### quantum/

**`qubo.py` — `TrafficQUBOBuilder`**
Builds the 6x6 QUBO matrix Q:
1. **Queue penalty** `w_queue`: diagonal term for heavier approach
2. **Coordination bonus** `w_coord`: off-diagonal reward for green-wave neighbors
3. **Spillback penalty** `w_spillback`: penalizes greening when downstream is near-full (>=80%)
4. **Emergency bias** `w_emergency=50`: forces green at ambulance path intersections

**`qaoa.py` — `QAOATrafficSolver`**
Manual PennyLane QAOA (no black-box operators):
1. Hadamard on all qubits -> uniform superposition
2. Cost layer (p=2): RZ for diagonal, CNOT-RZ-CNOT for off-diagonal
3. Mixer layer (p=2): RX on all qubits
4. COBYLA minimizes expected cost over 35 iterations
5. Top-4 most probable bitstrings evaluated; minimum-cost returned
6. Warm-start: optimal (gamma*, beta*) cached between rounds

**`ising.py` — `QUBOToIsingConverter`**
Converts QUBO `x^T Q x` to Ising `h_i sigma_i + J_ij sigma_i sigma_j` via `x_i = (1-sigma_i)/2`.

**`brute_force.py` — `BruteForceOptimizer`**
Exact: enumerates all 2^6=64 binary vectors, returns minimum QUBO cost assignment.

---

## Dashboard Pages & Visual Interface

The user interface is engineered with a high-contrast dark aesthetic, responsive glassmorphism, and distinct color-coded button navigation.

### Top Navigation & Telemetry Strips
- **Cleared Header**: Lowered main container (`padding-top: 4.2rem`) prevents collision with the Streamlit top navbar.
- **Emerald Title**: Styled with a vibrant mint/emerald linear gradient: `Quantum Traffic Brain`.
- **6 Emerald Metric Strips**: Dark-emerald cards (`#05261d` to `#0b3d2f`) with glowing mint metrics:
  - **Elapsed Time**: Simulation ticks elapsed.
  - **Avg Wait Time**: Mean waiting time per vehicle in seconds.
  - **Throughput (cpm)**: Completed vehicles per minute.
  - **Active Queue Depth**: Average queue accumulation across all approaches.
  - **Est. Idle Fuel (L\*)**: Estimated fuel consumption during stop-and-go idling (0.8 L/hr rate).
  - **Est. CO2 Emitted (kg\*)**: Estimated emissions based on fuel consumed (2.31 kg CO2/L).

---

### Navigation: 6 Full-Width Colored Button Tabs

The dashboard navigation uses full-width interactive button pills with clean gaps and alternating vibrant colors:

#### Tab 1: Traffic Orchestration Grid (Emerald Green)
- **Animated 2D Video Canvas (HTML/CSS/JS)**: Self-contained interactive 1100x560 px canvas running at 60 fps. Renders Bezier road curves connecting junctions A through F, dual-phase signal heads with realistic bloom glow, animated vehicle dots, emergency pods with flashing beacons, and blocked road hazard markers.
- **Diagnostic Canvas (Plotly)**: Interactive graph view supporting click-to-track vehicle inspection and link hazard toggling.
- **Perimeter Gate Queues & CCTV Media Detection**:
  - Left column: Direct vehicle injection inputs across all 10 boundary entry gates (N1–N3, S1–S3, W1–W2, E1–E2).
  - Right column: CCTV / Drone feed upload (JPG, PNG, WEBP, MP4) with instant preview and AI vehicle detection (YOLOv8n or 11-stage Classical CV fallback).
- **Intersection Queue Breakdown**: Live tabular breakdown of vehicular queues on N, S, E, W approaches per junction.
- **Dynamic Disruptions**: One-click simulation of traffic incidents and capacity restrictions.

#### Tab 2: Emergency Navigation Cockpit (Sapphire Medical Blue)
- **Tactical Paramedic HUD Header**: High-contrast command banner displaying CAD authority status (`108 EMS | QUANTUM CORRIDOR ACTIVE`).
- **`🏥 View Real Dashboard (Hospital Google Maps) ↗` Button**: Launches the dedicated full-page paramedic ambulance console in a **new browser tab** (`/?page=hospital_maps`).
- **Operator Authentication & Dispatch (Permanent on Main Page)**:
  - Driver Unit Call-Sign input (`Ambulance Unit 108`).
  - Starting Location selector (10 boundary gates N1–N3, S1–S3, W1–W2, E1–E2).
  - Destination Junction selector (Junctions A through F).
  - Preemption Mode: Soft QUBO Corridor bias ($W_{emerg}=15$) vs. Hard Override (forced green).
  - Priority Level: Priority 2 (Code Red / Cardiac) vs. Priority 1 (Code Yellow / Urgent).
  - Dispatch Controls: Single unit dispatch, Random auto-dispatch, and Multi-ambulance conflict resolution test dispatch.
  - Security Attack Simulator: Test harness for forged token attacks, expired token rejections, rate-limit burst blocks, and SHA-256 cryptographic audit chain verification.
  - Live Telemetry Canvas & Preemption Impact Monitor: Reports delay avoided (~9.7s to 13.4s) and downstream queue flushing.

#### Tab 3: Quantum QAOA Core (Electric Violet)
- **QUBO Matrix Heatmap**: Visualizes the $6 \times 6$ quadratic coupling and penalty matrix $Q$, illustrating coordination bonuses and spillback penalties.
- **QAOA State Probability Distribution**: Bar chart displaying the top 16 basis state probabilities measured from the PennyLane parameterized quantum circuit.
- **Circuit Telemetry**: Live readout of the QAOA approximation ratio, exact optimum hit status, and optimal phase bitstring.

#### Tab 4: Performance Benchmarks (Warm Amber)
- **On-Demand Benchmark Runner**: Executes synchronized multi-seed comparisons across all 4 controllers (Fixed-Timing Baseline, Greedy Rule-Based, Hybrid Brute-Force, and Hybrid QAOA).
- **Real-Time Performance Table**: Reports comparative throughput, average wait times, idle fuel savings, and baseline improvement percentages.

#### Tab 5: Evidence (Deep Teal)
- **20-Seed Independent Evaluation Benchmark**: Comprehensive multi-seed table evaluated across seeds 100–119 with 95% confidence intervals.
- **Preemption Pareto Trade-Off Study**: Empirical trade-off curve between ambulance time saved and civilian collateral delay across emergency bias weights $W_{emerg} \in [5, 150]$ vs Hard Override (`results/preemption_tradeoff.png`).
- **QAOA Algorithmic Depth ($p=1$ to $4$)**: Empirical approximation ratios across circuit depths (`results/qaoa_depth_vs_ratio.png`).
- **NISQ Depolarizing Noise Study**: Density matrix simulation of gate noise sensitivity (`results/qaoa_noise_study.png`).
- **Classical Brute-Force Combinatorial Scaling**: Measured $O(2^N)$ wall-clock runtime explosion from $N=4$ to $N=24$ (`results/scaling_curve.png`).

#### Tab 6: Security & Audit Log (Crimson / Ruby)
- **Emergency Green Corridor Cryptographic Governance**: Administrative management of CAD authentication tokens and access policies.
- **Append-Only SHA-256 Audit Log**: Interactive cryptographic ledger detailing every preemption request, granted corridor, and blocked attack.

---

### Standalone Page: Tactical EMS Paramedic Google Maps Console (`/?page=hospital_maps`)

Accessible via the **"🏥 View Real Dashboard (Hospital Google Maps) ↗"** button in Tab 2, this dedicated full-page interface provides genuine real-world navigation for ambulance operators:

```
+-----------------------------------------------------------------------------------+
| 🚑 CHENNAI EMS 108 — TACTICAL DISPATCH HUD       [⬅️ Switch Back to Traffic Brain] |
+---------------------------------------------------+-------------------------------+
|  1. Starting Location (From)                      |  REAL GOOGLE MAPS DISPLAY     |
|  [ Search Chennai address / landmark...         ] |                               |
|  [📍 Use My GPS Location]  [🗺️ Pick on Map]       |  - Roadmap / Satellite layers |
|  Quick: [Marina Beach] [T. Nagar] [Central] [Air] |  - Real road driving polyline |
|                                                   |  - Pulsing ambulance marker   |
|  2. Receiving Hospital (To)                       |  - Hospital trauma bay marker |
|  [ Apollo Hospitals (Greams Road)               v]|                               |
|  - Rajiv Gandhi Govt General Hospital             |  FLOATING HUD OVERLAY:        |
|  - TN Multi Super Speciality (Omandurar)          |  Speed: 58 km/h               |
|  - Kilpauk Medical College Hospital               |  Guidance: Follow Anna Salai  |
|  - MIOT International / Fortis Malar / Kauvery    |  Progress: 34%                |
|                                                   |                               |
|  3. Route Telemetry                               |                               |
|  Distance: 7.8 km  |  Est. Time: 11 mins          |                               |
|                                                   |                               |
|  [ 🚀 START EMERGENCY JOURNEY ]                   |                               |
+---------------------------------------------------+-------------------------------+
```

**Key Features:**
- **No Schematic Grid Jargon**: Completely replaces node IDs (`ja, jb, jc`) with genuine Chennai addresses, landmarks, and trauma hospitals.
- **Real Address Search**: Search any street or landmark powered by OpenStreetMap Nominatim, or choose quick preset chips (Marina Beach, T. Nagar Panagal Park, Central Railway Station, Guindy Kathipara, Chennai Airport).
- **HTML5 GPS Geolocation**: The **"Use My GPS Location"** button queries the browser's `navigator.geolocation` API to instantly pin the ambulance at your live physical coordinates.
- **Map-Click Pin Placement**: Click anywhere directly on the Google Map to drop or reposition pickup and hospital destinations.
- **OSRM Real Road Network Driving Paths**: Queries the Open Source Routing Machine (`router.project-osrm.org`) to compute the exact driving path along actual streets, calculating real road distance (km) and driving time.
- **Interactive Journey Simulation**: Clicking **"Start Emergency Journey"** smoothly animates the ambulance along the real road polyline with dynamic speed updates (55–68 km/h), arrival countdown, and turn-by-turn guidance.
- **Seamless Multitasking**: Operates in its own browser tab without disturbing the simulation state on the main dashboard.

---

## End-to-End System Flow

```
1. App starts -> st.session_state initialised
2. User clicks Auto-Simulate (N=15 steps)
   for each step:
     a. controller.compute_phases(sim)
        |- Build QUBO matrix Q from live queue state
        |- Run QAOA or brute-force -> bitstring x*
        +- Compute adaptive green durations
     b. sim.set_signal_phases(phases)
     c. sim.step()
        |- _generate_boundary_arrivals()  [if autoFlow]
        |- _discharge_queues()            [signal-gated]
        +- _advance_in_transit()
     d. emergency_mgr.advance_mission(sim)
     e. events_engine.tick(sim)
     f. st.rerun() -> canvas re-renders
3. Canvas HTML regenerated each rerun
   |- initial_vehicles from sim.queues + sim.in_transit
   |- signal_phases -> signal head colours
   |- ambulance path -> blue dot + green corridor
   +- accidents -> red X markers
```

---

## Quantum Optimization Pipeline

```
Traffic State (queue counts per approach)
         |
TrafficQUBOBuilder.build_qubo()
  |- Diagonal Q[i,i]: queue imbalance penalty
  |- Off-diagonal Q[i,j]: neighbor coordination bonus
  |- Spillback check: downstream saturation penalty
  +- Emergency bias: +50 weight for ambulance path nodes
         |
       Q matrix (6x6 float64)
         |
  QAOA mode:
    QUBOToIsingConverter -> h_i, J_ij
    PennyLane circuit (p=2 layers):
      |-- Hadamard init
      |-- Cost layer: RZ (diagonal), CNOT-RZ-CNOT (coupling)
      +-- Mixer layer: RX on all qubits
    COBYLA optimizer (35 iterations)
    -> optimal bitstring x* (6 bits)

  Brute-force mode:
    Enumerate all 64 binary vectors
    -> minimum cost bitstring x*
         |
  Phase map: {node_id: 0 or 1}  (NS/EW green)
         |
  Adaptive green duration:
  duration = clamp(15 + 0.8 * queue_active, 10, 45)
         |
  sim.set_signal_phases(phases)
```

---

## Emergency Corridor System

```
1. User clicks "Dispatch Ambulance"
   -> EmergencyCorridorManager.dispatch(origin, destination)
   -> Dijkstra weighted by (travel_time + queue_delay)
   -> Creates AmbulanceMission with path [A, B, C, F]

2. Each tick: emergency_mgr.advance_mission(sim)
   -> increments current_edge_progress_sec
   -> when progress >= edge_travel_time: advance to next node
   -> completed = True at final destination (hospital node)

3. HybridController.compute_phases() uses emergency_biases:
   -> For nodes where ambulance ETA < 45 s:
      force green in ambulance approach direction (weight=50)
   -> This dominates QUBO -> signal opens for ambulance

4. Canvas shows:
   -> Blue dot moving along route
   -> Green highlighted corridor edges
   -> Normal vehicles queue and wait at preempted signals

5. On completion: ambulance fades, normal optimization resumes
```

---

## Vehicle Detection Pipeline

```
User uploads file (JPG/PNG/WEBP/MP4)
         |
VehicleDetector.decode_media_bytes()
  |- Image: cv2.imdecode
  +- Video: cv2.VideoCapture -> sample at 25% position
         |
_get_yolo() -> load YOLOv8n from ~/.cache/ultralytics/
         |
If YOLO available:
  model.predict(img, conf=0.30, classes=[2,3,5,7])
  -> boxes, confidences, class IDs
  -> filter vehicle classes only

If YOLO unavailable (fallback):
  Classical 11-stage CV pipeline
  -> denoise -> CLAHE -> road mask -> Canny -> fuse
  -> morph-close -> dilate -> stripe-kill -> contour filter -> NMS

_annotate(img, detections, engine_name)
  -> colored reticle boxes per class (car=yellow, bus=blue...)
  -> HUD with engine tag + count

-> return (annotated_rgb, count, detections_list)
         |
map_detected_count_to_entries(count, gate)
         |
sim.inject_by_entry_name(gate, count)
-> vehicles appear on canvas at next rerun
```

---

## Configuration Parameters

All tunables reside in `traffic_quantum/config.py`:

| Parameter | Value | Effect |
|---|---|---|
| `grid_rows / grid_cols` | 2 / 3 | 6-junction urban topology (matches Chennai pilot corridor) |
| `base_arrival_rate` | 0.35 | Poisson probability of vehicle arrival per gate tick |
| `discharge_interval_ticks` | 2 | Ticks required to discharge leading vehicle through green signal |
| `default_capacity` | 20 | Maximum vehicle holding capacity per directed road segment |
| `w_queue` | 1.0 | QUBO linear weight on active queue depth imbalance |
| `w_coord` | 0.2 | QUBO quadratic coupling weight for arterial green-wave coordination |
| `w_spillback` | 0.5 | QUBO penalty for discharging into saturated downstream roads |
| `w_emergency` | 15.0 | QUBO emergency corridor bias for inbound ambulance approach |
| `w_pedestrian` | 0.4 | QUBO linear penalty for pedestrian crosswalk waiting queues |
| `pedestrian_arrival_rate` | 0.08 | Poisson arrival rate for crosswalk pedestrians per junction |
| `pedestrian_max_wait_sec` | 45 | Urgency threshold after which pedestrian phase is forced |
| `min_green_sec` | 10 | Minimum green duration enforced by Conflict Monitor |
| `reopt_interval_sec` | 10 | Interval (seconds) between successive QUBO re-optimizations |
| `p_layers` | 2 | QAOA circuit depth |
| `max_iterations` | 25 | Bounded classical optimizer (COBYLA) evaluation steps |
| `eta_threshold_sec` | 45.0 | Preemption anticipation horizon for incoming ambulances |
| `YOLO_CONF` | 0.30 | Minimum confidence threshold for vehicle bounding boxes |

---

## Defensible Multi-Seed Benchmark Results (20 Evaluation Seeds)

All evaluations were executed blindly across **20 independent evaluation seeds (seeds 100–119, 600 simulated seconds each)**. 
Hyperparameter tuning of QUBO weights ($W_{coord}, W_{spill}$) and re-optimization intervals was performed strictly on separate **training seeds (seeds 1–5)** to prevent overfitting.

### Benchmark Summary Table (Mean ± Std Dev & 95% Confidence Intervals)

| Controller | Avg Wait (s) | 95% Confidence Interval | Avg Ped Wait (s) | Throughput (cpm) | Avg Queue (cars) | Est. Fuel (L)* | Est. CO2 (kg)* | Ambulance Travel Time (s) | Extra Delay to Cross Traffic (s) | QAOA Approx Ratio | Exact Optimum Hit Rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Fixed-Timing Baseline** | 102.45 ± 4.37 | [100.54, 104.37] | 7.65 ± 0.74 | 138.4 ± 0.6 | 364.4 ± 19.7 | 47.01 ± 2.58 | 108.60 ± 5.96 | 15.3 ± 3.4 | 0.00 ± 0.00 | N/A | N/A |
| **Rule-Based (Longest Queue)** | 105.98 ± 5.93 | [103.38, 108.57] | 4.29 ± 0.65 | 131.0 ± 5.2 | 375.1 ± 23.8 | 48.96 ± 3.24 | 113.10 ± 7.49 | 21.7 ± 3.8 | 3.71 ± 3.55 | N/A | N/A |
| **Hybrid (Brute-Force)** | 106.50 ± 5.64 | [104.03, 108.98] | **3.37 ± 0.39** | 131.7 ± 3.9 | 377.6 ± 23.0 | 49.15 ± 3.08 | 113.54 ± 7.11 | **11.8 ± 2.2** | 4.11 ± 3.62 | N/A | N/A |
| **Hybrid (QAOA)** | 109.73 ± 5.15 | [107.47, 111.98] | 3.96 ± 0.45 | 130.1 ± 3.9 | 388.2 ± 20.9 | 50.70 ± 2.81 | 117.11 ± 6.50 | **8.8 ± 1.8** | 7.27 ± 3.88 | **0.9350 ± 0.0868** | **46.3%** |

\* *Fuel and CO2 are estimated from idle durations using standard EPA benchmarks (0.8 L/hr idle rate and 2.31 kg CO2/L petrol).*

### Honest Diagnosis: Why Does Hybrid Traffic Delay Differ from Fixed Baseline?
1. **Emergency & Pedestrian Prioritization Trade-off**:
   The hybrid controller actively prioritizes two critical societal safety metrics:
   - **Emergency response**: Ambulance travel time is slashed from 15.3s (Fixed) and 21.7s (Rule-Based) down to **11.8s (Brute-Force)** and **8.8s (QAOA)**.
   - **Pedestrian safety**: Pedestrian waiting time is cut by **56%** from 7.65s (Fixed) down to **3.37s (Brute-Force)** and 3.96s (QAOA).
   Holding cross-traffic to clear pedestrian crosswalks and open ambulance green corridors naturally imposes an average collateral delay of **+4.1s to +7.3s** on normal civilian vehicular traffic.
2. **Initial Underperformance Root Cause**:
   Prior to tuning, Hybrid (Brute-Force) suffered because high coordination ($W_{coord}=2.5$) and spillback ($W_{spill}=6.0$) weights overwhelmed the linear queue signal ($W_{queue}=1.0$), while a 30s decision interval allowed queues to pile up. Retuning to $W_{coord}=0.2, W_{spill}=0.5$ and re-optimizing every 10s restored responsiveness.
3. **QAOA Approximation Inaccuracy**:
   QAOA achieves an approximation ratio of $0.9350 \pm 0.0868$ with an exact optimum hit rate of $46.3\%$. Sub-optimal bitstrings chosen on ~54% of rounds under bounded COBYLA iterations slightly compound queue depth relative to exact Brute-Force ($388.2$ vs $377.6$ vehicles).

---

## Emergency Preemption Trade-Off: Soft QUBO vs Hard Override (Phase B)

A systematic sweep across 20 evaluation seeds (100–119) evaluated the Pareto trade-off between ambulance response time saved and collateral delay inflicted on cross-traffic:

| Preemption Policy | $W_{emerg}$ Weight | Ambulance Travel Time (s) | Time Saved (s) | Extra Delay on Normal Traffic (s) | Average Normal Wait (s) |
|---|---|---|---|---|---|
| **No Preemption** | 0.0 | 21.45 | 0.00 | 0.00 | 36.13 |
| **Soft QUBO Bias** | 5.0 | 12.00 | 9.45 | +1.15 | 37.28 |
| **Soft QUBO Bias** | **15.0** | **11.75** | **9.70** | **+1.38** | **37.51** |
| **Soft QUBO Bias** | 30.0 | 11.75 | 9.70 | +1.38 | 37.51 |
| **Soft QUBO Bias** | 50.0 | 11.75 | 9.70 | +1.38 | 37.51 |
| **Soft QUBO Bias** | 80.0 | 11.75 | 9.70 | +1.38 | 37.51 |
| **Soft QUBO Bias** | 150.0 | 11.75 | 9.70 | +1.38 | 37.51 |
| **Hard Override** | N/A (Forced Green) | 8.00 | 13.45 | +1.75 | 37.89 |

- **Key Takeaway**: Hard preemption clears the ambulance fastest (8.0s) but causes the greatest collateral disruption (+1.75s per vehicle across the network). Soft QUBO corridor preemption with $W_{emerg}=15$ achieves near-optimal response time (11.75s) while preserving network adaptability.
- Saved artifact: `results/preemption_tradeoff.png` and `results/preemption_tradeoff.json`.

---

## "Why Quantum?" Algorithmic Evidence (Phase E)

### 1. Simulated Annealing Baseline
Alongside Brute-Force and QAOA, a classical Simulated Annealing solver was evaluated on the traffic QUBO. Across 100 test states, Simulated Annealing finds optimal solutions in 98% of cases within 4.2ms, establishing a fast classical heuristic baseline for comparison against quantum annealing and gate-model QAOA.

### 2. QAOA Depth Study ($p=1$ to $4$)
Evaluated across 20 distinct traffic snapshots with bounded classical iterations (25 COBYLA steps):
- **$p=1$**: Approximation ratio $0.9129 \pm 0.0935$, Exact hit rate: 40%
- **$p=2$**: Approximation ratio **$0.9517 \pm 0.0714$**, Exact hit rate: **60%** (Optimal trade-off)
- **$p=3$**: Approximation ratio $0.9327 \pm 0.0798$, Exact hit rate: 50%
- **$p=4$**: Approximation ratio $0.9220 \pm 0.0668$, Exact hit rate: 30%
- *Observation*: Without exponential classical optimization budget, higher circuit depths suffer from barren plateaus and optimizer convergence limits on finite-step COBYLA.
- Saved artifact: `results/qaoa_depth_vs_ratio.png`.

### 3. NISQ Depolarizing Noise Study
Simulated on PennyLane's `default.mixed` density matrix simulator under single-qubit depolarizing noise:
- **Ideal (Noiseless)**: Ratio $0.7658 \pm 0.1522$ (density matrix statevector)
- **Low Noise ($p_{gate}=0.005$)**: Ratio $0.8081 \pm 0.1311$
- **Medium Noise ($p_{gate}=0.02$)**: Ratio $0.7440 \pm 0.1353$
- **High Noise ($p_{gate}=0.05$)**: Ratio $0.7991 \pm 0.1412$
- Saved artifact: `results/qaoa_noise_study.png`.

### 4. Classical Combinatorial Scaling ($O(2^N)$ Explosion)
Brute-force exhaustive search runtime measured from $N=4$ to $N=20$ intersections ($2^N$ states) and extrapolated to $N=24$:

| Intersections ($N$) | Search Space ($2^N$ states) | Classical Wall-Clock Runtime | Evaluations/sec |
|---|---|---|---|
| 4 | 16 | 0.00013 s | 120,663 |
| 6 | 64 | 0.00028 s | 231,297 |
| 8 | 256 | 0.00114 s | 225,332 |
| 10 | 1,024 | 0.00509 s | 201,139 |
| 12 | 4,096 | 0.04293 s | 95,403 |
| 14 | 16,384 | 0.08939 s | 183,285 |
| 16 | 65,536 | 0.33317 s | 196,707 |
| 18 | 262,144 | 1.44147 s | 181,858 |
| 20 | 1,048,576 | 7.61289 s | 137,736 |
| 22 | 4,194,304 | 24.37 s (projected) | 172,100 |
| 24 | 16,777,216 | 97.48 s (projected) | 172,100 |

- *Observation*: Classical brute force is viable at $N=6$ (<1ms), but at $N=24$ it requires over 1.5 minutes per single re-optimization step, rendering real-time adaptive control impossible without quantum or meta-heuristic formulations.
- Saved artifact: `results/scaling_curve.png` and `results/scaling_table.csv`.

---

## Hardware-Ready Architecture (Phase G)

To decouple optimization logic from physical actuator dynamics, a vendor-agnostic interface layer is introduced in `traffic_quantum/signal_interface.py`:

```python
class SignalControllerInterface(ABC):
    @abstractmethod
    def set_phase(self, intersection_id: int, phase: int, duration: float) -> bool: ...
    @abstractmethod
    def get_state(self, intersection_id: int) -> Dict[str, Any]: ...
    @abstractmethod
    def emergency_preempt(self, corridor_nodes: List[int], phases: Dict[int, int]) -> bool: ...
    @abstractmethod
    def safe_fallback(self) -> None: ...
```

### Safety Features Implemented
1. **Software Conflict Monitor (`SoftwareConflictMonitor`)**:
   - Strictly prohibits conflicting simultaneous green indications (e.g. concurrent N-S and E-W greens).
   - Enforces a **10-second minimum green constraint** to prevent rapid flickering.
   - Enforces a **2-second yellow clearance interval** on all phase transitions.
2. **Hardware Watchdog & Heartbeat Monitor (`HardwareWatchdog`)**:
   - Monitors controller pulse heartbeats. If quantum or classical optimization times out or exceeds its execution budget (>15s elapsed), the watchdog immediately triggers `safe_fallback()` into fail-safe fixed-timing cycle.

### Future Work: Physical Signal Hardware Integration
The `SignalControllerInterface` abstracts the underlying physical signal mechanism. In future deployments, this software layer enables seamless substitution of the `TrafficSimulator` with real-world **NEMA TS2**, **Type 170**, or **ATC (Advanced Transportation Controller)** cabinet field controllers using standard **NTCIP 1202** communications protocols without altering any QUBO or QAOA code. 
*Note: No physical hardware is currently integrated or connected; all validations are conducted in simulation.*

---

## Security Architecture & Spoofing Defense (Phase F)

### Security Safeguards
- **HMAC-SHA256 JWT Authentication**: All preemption dispatches require a cryptographically signed token with role-based claims (`emergency_vehicle`, `transit_priority`).
- **Rate-Limiting Protection**: Token-bucket limiter restricts emergency preemption requests to a maximum of 5 requests per 30 seconds per vehicle ID, neutralizing denial-of-service spam.
- **Append-Only SHA-256 Hash Chaining**: Every dispatch, rejection, and preemption override is logged to an immutable audit chain (`audit_log.jsonl`). The integrity verifier re-computes the entire chain hash to detect unauthorized tampering.
- **Interactive Security Attack Demo**: The Driver Cockpit includes a dedicated test harness that simulates forged token attacks, expired token attempts, and rate-limit bursts, demonstrating real-time cryptographic rejection.
- *Notice*: In this interactive prototype, the JWT issuer is collocated within the Streamlit dashboard for demonstration purposes. In a real-world municipal deployment, tokens are issued exclusively by an isolated, air-gapped Computer-Aided Dispatch (CAD) municipal authority.

---

## Assumptions vs. Empirical Measurements

To maintain complete scientific integrity, all metrics in this system are strictly classified into either **empirical measurements** or **configurable baseline assumptions**:

| Metric / Parameter | Category | Basis / Citation |
|---|---|---|
| Vehicle Average Wait Time | **Empirical Measurement** | Measured from tick-level vehicle queue dwell timers across 20 evaluation seeds. |
| Pedestrian Average Wait Time | **Empirical Measurement** | Measured from individual pedestrian crosswalk queue timers across 20 seeds. |
| Network Throughput | **Empirical Measurement** | Measured total vehicles successfully traversing perimeter exit boundaries per minute. |
| Active Queue Depth | **Empirical Measurement** | Measured instantaneous vehicle counts accumulated behind stop lines. |
| Ambulance Response Time | **Empirical Measurement** | Measured tick difference between ambulance dispatch and hospital node arrival. |
| QAOA Approximation Ratio | **Empirical Measurement** | Ratio of QAOA expectation value to exact Brute-Force ground truth QUBO minimum. |
| Exact Optimum Hit Rate | **Empirical Measurement** | Percentage of runs where sampled QAOA bitstring matches the exact global minimum. |
| Idle Fuel Consumption Rate | **Configurable Assumption** | EPA standard assumption: 0.8 L/hr for idling internal combustion passenger vehicles. |
| Fuel-to-CO2 Conversion Rate | **Configurable Assumption** | EPA standard assumption: 2.31 kg CO2 per liter of gasoline consumed. |
| Road Segment Capacity | **Configurable Assumption** | Fixed geometric model assumption: 20 passenger cars per 200m directed road link. |
| Urban Speed Limit | **Configurable Assumption** | Standard schematic model assumption: 45 km/h (12.5 m/s) free-flow travel speed. |

---

## Installation & Running

### Prerequisites
- Python 3.11+
- CUDA GPU optional (YOLOv8 runs smoothly on CPU)

### Install
```bash
git clone https://github.com/your-repo/hack-quant.git
cd hack-quant

pip install -r traffic_quantum/requirements.txt
```

### Environment Configuration (.env)
Copy the example environment file and set local credentials:
```bash
cp .env.example .env
```
The application defaults to open-source **OpenStreetMap / CartoDB** tiles with zero proprietary API dependencies. Google Maps tiles are available behind an explicit "demo only" UI toggle.

### Run Dashboard
```bash
streamlit run dashboard.py
# Access dashboard at http://localhost:8501
```

### Run Multi-Seed Benchmark Suite
```bash
python traffic_quantum/benchmark.py
```

### Run Full Test Suite
```bash
pytest traffic_quantum/tests -v
```

---

## Known Limitations

| Area | Limitation | Mitigation / Planned Resolution |
|---|---|---|
| **Simulator Scale** | 2x3 grid (6 intersections / 6 qubits) simulated locally on CPU. | Sufficient to demonstrate quantum encoding; physical QPUs or tensor networks required for >30 qubits. |
| **QAOA CPU Latency** | 6-qubit QAOA takes ~0.5–1.0s per solve on CPU. | For live interactive UI testing, Brute-Force mode provides instantaneous (<1ms) solving. |
| **Vehicle Detection** | YOLOv8n is an edge nano model; low-angle camera occlusions can cause vehicle under-counting. | Classical CV morphological fallback pipeline ensures detection continuity even without YOLO weights. |
| **Preemption Delay** | Emergency green corridors increase cross-street vehicle queue wait times. | Soft preemption ($W_{emerg}=15$) balances corridor speed against cross-street gridlock. |
| **Hardware Link** | Physical traffic cabinet controllers are simulated via software abstraction. | The `SignalControllerInterface` is designed for direct drop-in integration with NTCIP 1202 controller hardware. |

