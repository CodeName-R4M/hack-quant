"""Phase E.3: QAOA Noise Study on PennyLane default.mixed.

Evaluates how quantum depolarizing noise degrades the QAOA approximation ratio.
Simulates noisy hardware execution across 3 depolarizing error rates:
lambda in [0.005, 0.02, 0.05], comparing against ideal noiseless simulation.

Outputs results to results/qaoa_noise_study.png and results/qaoa_noise_data.json.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from traffic_quantum.config import DEFAULT_CONFIG
from traffic_quantum.network import RoadNetwork
from traffic_quantum.quantum.brute_force import BruteForceOptimizer
from traffic_quantum.quantum.ising import QUBOToIsingConverter
from traffic_quantum.quantum.qubo import TrafficQUBOBuilder
from traffic_quantum.simulator import TrafficSimulator


def run_noisy_qaoa_instance(Q: np.ndarray, C0: float, noise_rate: float, p: int = 2) -> float:
    """Runs QAOA on PennyLane's default.mixed with depolarizing noise channels."""
    n = Q.shape[0]
    h, J, offset = QUBOToIsingConverter.qubo_to_ising(Q, C0)
    exact_x, exact_cost, _ = BruteForceOptimizer.solve(Q, C0)

    # Precompute costs for all 2^n states
    all_x = np.zeros((1 << n, n), dtype=np.int32)
    for i in range(1 << n):
        for bit in range(n):
            all_x[i, n - 1 - bit] = (i >> bit) & 1

    costs = np.array([TrafficQUBOBuilder.evaluate_qubo(Q, C0, all_x[idx]) for idx in range(1 << n)])

    if noise_rate > 0.0:
        dev = qml.device("default.mixed", wires=n)
    else:
        dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="autograd")
    def noisy_circuit(angles):
        gamma = angles[0]
        beta = angles[1]

        # Equal superposition
        for w in range(n):
            qml.Hadamard(wires=w)
            if noise_rate > 0.0:
                qml.DepolarizingChannel(noise_rate, wires=w)

        # QAOA layers
        for layer in range(p):
            g = gamma[layer]
            b = beta[layer]

            # Cost layer
            for i in range(n):
                if abs(h[i]) > 1e-7:
                    qml.RZ(2.0 * g * h[i], wires=i)
                    if noise_rate > 0.0:
                        qml.DepolarizingChannel(noise_rate, wires=i)

            for i in range(n):
                for j in range(i + 1, n):
                    if abs(J[i, j]) > 1e-7:
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2.0 * g * J[i, j], wires=j)
                        qml.CNOT(wires=[i, j])
                        if noise_rate > 0.0:
                            qml.DepolarizingChannel(noise_rate, wires=j)

            # Mixer layer
            for i in range(n):
                qml.RX(2.0 * b, wires=i)
                if noise_rate > 0.0:
                    qml.DepolarizingChannel(noise_rate, wires=i)

        return qml.probs(wires=range(n))

    # Optimize angles with COBYLA
    def obj(flat_params):
        probs = np.array(noisy_circuit(flat_params.reshape((2, p))), dtype=np.float64)
        return float(np.dot(probs, costs))

    init_params = np.concatenate([np.linspace(0.1, 0.4, p), np.linspace(0.4, 0.1, p)])
    res = minimize(obj, init_params, method="COBYLA", options={"maxiter": 20})

    final_probs = np.array(noisy_circuit(res.x.reshape((2, p))), dtype=np.float64)
    best_idx = int(np.argmax(final_probs))
    best_cost = costs[best_idx]

    min_c = np.min(costs)
    max_c = np.max(costs)
    rng_c = max_c - min_c if max_c > min_c else 1.0
    approx_ratio = float((max_c - best_cost) / rng_c)
    return max(0.0, min(1.0, approx_ratio))


def run_noise_study():
    """Evaluates noise impact on QAOA approximation ratio."""
    print("Starting QAOA Noise Degradation Study on default.mixed...")
    net = RoadNetwork()
    builder = TrafficQUBOBuilder(net)

    # 10 test snapshots
    snapshots = []
    for s in range(10):
        sim = TrafficSimulator(net, seed=s + 50)
        for _ in range((s + 1) * 12):
            sim.step({n: (s + n) % 2 for n in net.graph.nodes})
        Q, C0 = builder.build_qubo(sim)
        snapshots.append((Q, C0))

    noise_levels = [0.0, 0.005, 0.02, 0.05]
    labels = ["Ideal (Noiseless)", "Low (p=0.005)", "Medium (p=0.02)", "High (p=0.05)"]
    noise_results = {}

    for noise, lbl in zip(noise_levels, labels):
        print(f"Running Noise Level: {lbl}...")
        ratios = []
        for Q, C0 in snapshots:
            r = run_noisy_qaoa_instance(Q, C0, noise_rate=noise, p=2)
            ratios.append(r)
        mean_r = float(np.mean(ratios))
        std_r = float(np.std(ratios))
        noise_results[lbl] = {
            "noise_rate": noise,
            "mean_ratio": round(mean_r, 4),
            "std_ratio": round(std_r, 4),
        }
        print(f"  {lbl}: Mean Approx Ratio = {mean_r:.4f} +/- {std_r:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/qaoa_noise_data.json", "w") as f:
        json.dump(noise_results, f, indent=2)

    # Plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#111827")

    x_indices = np.arange(len(noise_levels))
    means = [noise_results[l]["mean_ratio"] for l in labels]
    stds = [noise_results[l]["std_ratio"] for l in labels]
    bar_colors = ["#22c55e", "#38bdf8", "#f59e0b", "#ef4444"]

    bars = ax.bar(x_indices, means, yerr=stds, capsize=6, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=1.2, width=0.55)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m / 2, f"{m:.3f}", ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(labels, color="#cbd5e1", fontsize=10)
    ax.set_ylabel("Approximation Ratio $\\alpha$", color="#cbd5e1", fontsize=11, labelpad=8)
    ax.set_ylim(0.0, 1.15)
    ax.grid(True, linestyle=":", alpha=0.3, color="#475569", axis="y")
    ax.tick_params(colors="#94a3b8")

    plt.title("QAOA Robustness Under Depolarizing Noise (PennyLane default.mixed)\np=2 Layers, 6 Intersections, 10 Snapshots", color="#f8fafc", fontsize=11, pad=12)
    plt.tight_layout()
    chart_path = "results/qaoa_noise_study.png"
    plt.savefig(chart_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Noise chart saved to {chart_path} and data to results/qaoa_noise_data.json")
    return noise_results


if __name__ == "__main__":
    run_noise_study()
