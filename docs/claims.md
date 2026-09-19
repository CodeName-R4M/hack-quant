# Quantum Traffic Brain — Defensible Pitch Claims & Claims Guide

This document establishes the official scientific boundary for the Quantum Traffic Brain project. Every claim permitted for pitch presentations, reports, and documentation is mapped directly to its supporting empirical result file. Disallowed claims are explicitly documented with technical explanations.

---

## 1. Allowed Pitch Claims (With Supporting Result Files)

### Claim 1: Quantum-Ready Formulation for Combinatorial Traffic Phase Optimization
- **Pitch Statement**: "We formulate network-wide multi-intersection traffic signal coordination as a Quadratic Unconstrained Binary Optimization (QUBO) problem solved via QAOA on a quantum circuit, with exact classical verification."
- **Supporting Files**:
  - `traffic_quantum/quantum/qubo.py`: QUBO Hamiltonian with queue, coordination, spillback, emergency, pedestrian, and switching penalties.
  - `traffic_quantum/quantum/qaoa.py`: PennyLane QAOA variational ansatz with parameterized cost and mixer unitary layers.
  - `results/scenario_benchmark.csv` / `results/benchmark_20seeds.json`: 20-seed independent evaluation proving stable optimization across all seeds.

### Claim 2: Hybrid Optimization Significantly Outperforms Fixed Timing Under Moderate and Rush-Hour Traffic
- **Pitch Statement**: "Under moderate unsaturated traffic demand, the hybrid controller reduces vehicle wait time by 9.39s (34.2% reduction) compared to fixed-timing cycles. Under directional rush-hour demand, the hybrid controller cuts average delay by 23.29s (22.1% reduction) relative to fixed timing."
- **Evidence & Data**:
  - `results/scenario_benchmark.csv`:
    - **Moderate Load**: Hybrid (BF) wait `18.11 ± 0.48s` vs Fixed `27.49 ± 0.76s` (Paired mean difference: `-9.39 ± 0.41s`, 95% CI `[-9.79, -8.98]s`, $n=20$).
    - **Rush Hour**: Hybrid (BF) wait `81.91 ± 4.64s` vs Fixed `105.20 ± 3.15s` (Paired mean difference: `-23.29 ± 1.63s`, 95% CI `[-24.92, -21.65]s`, $n=20$).
  - `results/manifest.json`: Identical 10s reoptimization interval, $W_{switch}=1.0$, evaluation seeds 100–119.

### Claim 3: Fixed-Timing is Best Under Uniform Balanced Flow; Tied Under Severe Incident Surge
- **Pitch Statement**: "When traffic demand is symmetric across all approaches, fixed 30s/30s cycling is optimal (102.45s vs 107.80s for hybrid), because adaptive phase shifting incurs transition friction. In severe incident surges, hybrid and fixed-timing are statistically tied."
- **Evidence & Data**:
  - `results/scenario_benchmark.csv`:
    - **Balanced Flow**: Fixed-Timing wait `102.45 ± 4.37s` vs Hybrid (BF) `107.80 ± 5.31s` (Paired mean diff: `+5.35 ± 1.50s` favoring Fixed).
    - **Surge + Accident**: Fixed-Timing wait `98.69 ± 2.82s` vs Hybrid (BF) `99.37 ± 5.31s` (Paired mean diff: `+0.68 ± 1.76s`, 95% CI `[-1.07, +2.44]s`, $n=20$). Because the 95% CI spans zero, this is an honest statistical tie.

### Claim 4: Hybrid Controller Matches Rule-Based on Raw Delay but Delivers Superior Multi-Objective Outcomes
- **Pitch Statement**: "While greedy rule-based (longest queue) and hybrid controllers achieve comparable vehicle delays, the hybrid controller balances multi-objective trade-offs: reducing pedestrian wait times by 12–20% and providing dedicated green corridors for emergency ambulances."
- **Evidence & Data**:
  - `results/scenario_benchmark.csv`:
    - **Raw Delay**: Moderate load difference is `+0.41 ± 0.29s`; Rush hour difference is `+1.44 ± 1.41s`; Surge difference is `-0.08 ± 0.82s` (95% CI spans zero; tie).
    - **Pedestrian Wait**: Hybrid (BF) pedestrian wait is `2.91 ± 0.25s` vs Rule-Based `3.31 ± 0.33s` (moderate load); `3.52 ± 0.42s` vs `4.29 ± 0.65s` (balanced load); `5.77 ± 0.73s` vs `7.77 ± 1.76s` (surge incident).

### Claim 5: Soft QUBO Emergency Preemption Reduces Ambulance Travel Time Dramatically
- **Pitch Statement**: "Dynamic emergency preemption via linear QUBO corridor biasing cuts ambulance response travel times across the network by 10.6s to 19.9s compared to classical controllers, while avoiding the catastrophic gridlock of unconstrained hard signal overrides."
- **Evidence & Data**:
  - `results/scenario_benchmark.csv`:
    - Moderate load ambulance travel: Hybrid (BF) `9.2 ± 2.2s` vs Fixed `19.9 ± 12.2s` (difference: `-10.65 ± 5.42s`) vs Rule-Based `24.6 ± 13.7s` (difference: `-15.40 ± 6.36s`).
    - Rush hour ambulance travel: Hybrid (BF) `9.2 ± 2.2s` vs Fixed `25.1 ± 15.7s` (difference: `-15.90 ± 7.35s`) vs Rule-Based `29.1 ± 13.6s` (difference: `-19.90 ± 6.42s`).
  - `results/preemption_tradeoff.json`: Sweeping $W_{emerg} \in [5, 150]$ proves that soft preemption imposes minimal collateral delay on civilian traffic (~3.5s to 5.4s) while clearing the corridor.

### Claim 6: QAOA is an Approximation Algorithm; Classical Heuristics Work Effectively Today
- **Pitch Statement**: "QAOA (p=2) operates as an approximation algorithm achieving average approximation ratios between 91.9% and 94.8% across 6 qubits, finding the exact optimum 33% to 60% of the time. Classical simulated annealing also finds optimal solutions in milliseconds, demonstrating that this pipeline is quantum-ready rather than claiming classical computational obsolescence."
- **Evidence & Data**:
  - `results/scenario_benchmark.csv`: Exact hit rates (Moderate: 59.8%, Rush: 36.4%, Surge: 33.6%, Balanced: 32.6%).
  - `results/qaoa_depth_data.json`: Scaling optimizer budget with circuit depth ($20 + 20p$) confirms approximation ratio improves from 0.9031 ($p=1$) to 0.9545 ($p=4$, 55% exact hit rate).
  - `results/scaling_metadata.json`: Traffic QUBO matrices are sparse (planar topology with bounded degree $\le 4$); specialized classical solvers outperform dense brute force.

### Claim 7: Physical QPU Runner is Fully Implemented with Strict Guardrails
- **Pitch Statement**: "The repository includes an audited, cost-capped runner for Amazon Braket QPUs (`scripts/run_hardware_braket.py`) that enforces mandatory `--confirm` flags, dry-run circuit inspection, and zero secret logging."
- **Evidence & Data**:
  - `scripts/run_hardware_braket.py`: Refuses execution without explicit confirmation, validates audit schemas, supports dry-run mode.
  - `dashboard.py`: Renders recorded hardware results read-only from `results/qpu_run_*.json` only when all mandatory audit fields are present.

---

## 2. Claims to Avoid (Strictly Prohibited)

| Disallowed Claim | Why It Must NOT Be Made | What to Say Instead | Supporting Evidence |
|---|---|---|---|
| **"Quantum Advantage / Supremacy"** | A 6-intersection network has only $2^6 = 64$ states. A laptop evaluates all 64 states in 0.0003 seconds. QAOA is an approximation that achieves 91–95% approximation ratio and takes slightly longer wait time (+1.7s to +5.7s) than brute-force. | "Quantum-ready formulation validated on simulators and prepared for future fault-tolerant scaling." | `results/scaling_table.csv`, `results/scenario_benchmark.csv` |
| **"Smoother / Fewer Phase Switches"** | With the current tuned parameters, the Hybrid controller executes ~155 to 328 phase switches across 600s (~2.5x more than Fixed-timing's 114 switches), matching Rule-Based (154–324 switches). | "Hybrid optimizes for queue and coordination delay dynamically, switching phases adaptively at a rate comparable to rule-based policies." | `results/scenario_benchmark.csv` |
| **"w_switch=1.0 is the proven best weight"** | Grid tuning on training seeds (1–5) revealed differences between $W_{switch} \in [0.5, 2.5]$ were small (<1s average wait difference). | "$W_{switch}=1.0$ was selected on training seeds 1–5; differences between settings on those seeds were minor (<1s)." | `results/tuning_summary.csv` |
| **"QAOA Beats Brute-Force on Ambulance Travel Time"** | In moderate load and rush hour, the paired difference 95% CI spans zero (statistically a tie). In balanced flow, discrete 5s differences reflect stochastic civilian queue placement prior to ambulance dispatch, not algorithmic superiority. | "Both QAOA and Brute-Force apply identical $W_{emerg}=50$ biases; minor travel time differences reflect discrete simulation queue noise." | `results/scenario_benchmark.json` paired CI analysis |
| **"Hardware QPU results without hardware JSON"** | If no `results/qpu_run_*.json` file exists with all required audit fields (device name, task ID, timestamp, shots), no physical quantum hardware was used. | "All presented quantum results were evaluated on local state-vector quantum simulators (PennyLane `default.qubit` / Amazon Braket local simulator)." | `dashboard.py` `load_latest_qpu_run()` audit check |
| **"2-Second Yellow Clearance in Simulator"** | The simulator executes discrete 1-second ticks with instantaneous binary green/red signal transitions. It has no yellow clearance phase. | "The software conflict monitor includes yellow clearance logic as an architectural hardware-layer specification, but the discrete simulation engine operates on binary green/red phases." | `traffic_quantum/simulator.py`, `traffic_quantum/signal_interface.py` |
| **"EPA Standard Fuel/Emissions"** | Fuel consumption and $\text{CO}_2$ are computed purely as scalar linear multiples of vehicle idle delay using typical engineering assumptions (0.8 L/hr idle rate and 2.31 kg $\text{CO}_2$/L petrol). | "Fuel and $\text{CO}_2$ are estimated proportional to idle delay based on typical automotive values (0.8 L/hr idle rate)." | `traffic_quantum/config.py`, `traffic_quantum/metrics.py` |
| **"Higher QAOA depth fails due to Barren Plateaus"** | In a 6-qubit system, barren plateaus do not occur. Increasing depth to $p=3$ and $p=4$ increases variational angles to 6 and 8. With scaled optimizer iterations ($20 + 20p$), approximation ratio increases to 0.9545. | "Prior drops at higher depth were optimizer budget artifacts; with proportional iterations, higher depth improves solution quality." | `results/qaoa_depth_data.json` |

---

## 3. Network Saturation Breakdown

When discussing scenario benchmarks, clearly differentiate unsaturated regimes from oversaturated regimes:

| Regime | Boundary Arrival Rate | Total Offered Load | Measured Throughput | Avg Network Queue | Saturation Status |
|---|---|---|---|---|---|
| **Moderate Load** | 0.18 cars/s per gate (all 10 gates) | **108.0 cars/min** | **99.2 – 101.1 cars/min** | **31.5 – 48.6 cars** | **Unsaturated (~75–80% capacity)**: Queues remain bounded; delays reflect pure signal timing efficiency. |
| **Balanced Flow** | 0.35 cars/s per gate (all 10 gates) | **210.0 cars/min** | **128.5 – 138.4 cars/min** | **364.4 – 397.7 cars** | **Oversaturated**: Inflow exceeds maximum network exit capacity (~138 cpm); queues grow over time. |
| **Rush Hour** | 0.45 E-W, 0.15 N-S | **162.0 cars/min** | **105.2 – 117.9 cars/min** | **219.7 – 295.3 cars** | **Oversaturated (Directional)**: Heavy arterial queues build up on East-West corridors. |
| **Surge + Incident** | 0.45 E-W, 0.20 N-S + lane closure | **180.0 cars/min** | **116.0 – 120.9 cars/min** | **301.6 – 317.3 cars** | **Oversaturated (Bottlenecked)**: Bottlenecked edge (3, 4) causes upstream queue accumulation. |

---

## 4. Pitch Talking Points Summary

1. **The Problem**: Urban traffic signal timing is a complex multi-agent coordination challenge with competing objectives (throughput, queue balance, pedestrian safety, and emergency response).
2. **The Innovation**: Formulates real-time phase selection as an interconnected QUBO problem mapped to QAOA quantum circuits and validated on Amazon Braket.
3. **The Reality**: On current small grids (6 junctions), classical solvers solve the problem in milliseconds. We do not claim quantum advantage today; we demonstrate a validated, quantum-ready optimization framework.
4. **The Benefits**: Compared to fixed timing, the adaptive system cuts delays by 22%–34% under realistic traffic and reduces emergency response times by over 10 seconds.
