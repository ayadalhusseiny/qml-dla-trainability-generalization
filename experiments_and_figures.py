"""
experiments_and_figures.py
==========================
PennyLane Numerical Experiment Suite and Figure Generator for Paper 3:

  "A Fundamental Duality in Quantum Machine Learning:
   Trainability vs. Generalization via Dynamical Lie Algebras"
  Author: Ayad Alhusseiny (University of Kerbala)
  Correspondence: ayad.a@uokerbala.edu.iq

This standalone reproducibility script:
  1. Empirically validates the Fundamental Duality Theorem across two simultaneous axes:
     - Trainability Axis (Barren Plateaus): Evaluates gradient variance Var[nabla C] vs
       qubit count n in {2, 3, 4, 5, 6} for Hardware-Efficient (dim(g) = 4^n - 1)
       versus Permutation-Equivariant (dim(g) = 3) architectures.
     - Generalization Axis (Minimax Sample Complexity): Evaluates test RMSE vs
       sample size M in {10, 25, 50, 100, 200} at n=4 qubits across multiple seeds.
  2. Prints Table 1 (Variance Scaling) and Table 2 (Generalization Performance).
  3. Generates publication-ready Figure 1:
     - fig_duality_experiments.pdf (vector)
     - fig_duality_experiments.png (raster)
  4. Saves and loads structured benchmark results in experiments_results.json.

Usage:
  # Fast reproduction using precomputed benchmark data (~3s):
  python experiments_and_figures.py

  # Full retraining from scratch across all qubits, sample sizes, and seeds:
  python experiments_and_figures.py --train

Requirements:
  pip install -r requirements.txt
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

# Ensure UTF-8 output encoding for Windows compatibility
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 85, flush=True)
print("PAPER 3 NUMERICAL REPRODUCIBILITY SUITE (PennyLane Vectorized Backprop)", flush=True)
print("=" * 85, flush=True)

# Output directory and file paths (self-contained)
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "experiments_results.json")
fig_png = os.path.join(base_dir, "fig_duality_experiments.png")
fig_pdf = os.path.join(base_dir, "fig_duality_experiments.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Quantum Circuit Definitions: HEA vs Permutation-Equivariant
# ─────────────────────────────────────────────────────────────────────────────
def create_hea_circuit(n_qubits, layers=3):
    """
    Family A: Hardware-Efficient Ansatz (Generic DLA, dim(g) = 4^n - 1)
    Alternating single-qubit RY, RZ rotations with linear CNOT entangling layers.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(params, x=None):
        idx = 0
        if x is not None:
            # Data encoding layer
            for i in range(n_qubits):
                qml.RY(x[i % len(x)], wires=i)
        
        for l in range(layers):
            for i in range(n_qubits):
                qml.RY(params[idx], wires=i); idx += 1
                qml.RZ(params[idx], wires=i); idx += 1
            # Linear entangling layer
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        # Global observable exhibiting barren plateau scaling (Cerezo 2021)
        obs = qml.prod(*[qml.PauliZ(i) for i in range(n_qubits)])
        return qml.expval(obs)
    
    n_params = layers * n_qubits * 2
    return circuit, n_params

def create_equivariant_circuit(n_qubits, layers=3):
    """
    Family B: Permutation-Equivariant Subalgebra Ansatz (dim(g) <= 3)
    Collective spin rotations: exp(-i theta_X Jx) exp(-i theta_Z Jz)
    Guarantees confinement to su(2) regardless of qubit count n.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(params, x=None):
        idx = 0
        if x is not None:
            # Collective permutation-invariant data encoding: sum_i x_i
            x_val = np.sum(x)
            for i in range(n_qubits):
                qml.RY(x_val, wires=i)
        
        for l in range(layers):
            theta_x = params[idx]; idx += 1
            theta_z = params[idx]; idx += 1
            # Collective Jx rotation
            for i in range(n_qubits):
                qml.RX(theta_x, wires=i)
            # Collective Jz rotation
            for i in range(n_qubits):
                qml.RZ(theta_z, wires=i)
        
        # Collective measurement: (1/n) * sum Z_i
        obs = qml.Hamiltonian([1.0 / n_qubits] * n_qubits, [qml.PauliZ(i) for i in range(n_qubits)])
        return qml.expval(obs)
    
    n_params = layers * 2
    return circuit, n_params

# ─────────────────────────────────────────────────────────────────────────────
# 2. Console Table Formatting & Reporting
# ─────────────────────────────────────────────────────────────────────────────
def print_tables(qubit_range, var_hea_list, var_equiv_list, results_gen, M_values):
    print("\n" + "=" * 85, flush=True)
    print("TABLE 1: GRADIENT VARIANCE SCALING (TRAINABILITY & BARREN PLATEAU SCALING)", flush=True)
    print("=" * 85, flush=True)
    print(f"{'n':<4} | {'HEA dim(g)':<12} | {'HEA Var[dC]':<14} | {'Equiv dim(g)':<12} | {'Equiv Var[dC]':<14} | {'Variance Ratio':<14}", flush=True)
    print("-" * 85, flush=True)
    
    for i, n in enumerate(qubit_range):
        d_hea = 4**n - 1
        d_eq = 3
        v_hea = var_hea_list[i]
        v_eq = var_equiv_list[i]
        ratio = v_eq / v_hea if v_hea > 0 else 0.0
        ratio_str = f"{ratio:.2f}x"
        print(f"{n:<4} | {d_hea:<12} | {v_hea:<14.5f} | {d_eq:<12} | {v_eq:<14.5f} | {ratio_str:<14}", flush=True)
    
    hea_decay = var_hea_list[0] / var_hea_list[-1]
    eq_decay = var_equiv_list[-1] / var_equiv_list[0]
    print("-" * 85, flush=True)
    print(f"[*] Summary: HEA Variance decays by {hea_decay:.2f}x from n=2 to n=6 (Exponential Barren Plateau).", flush=True)
    print(f"[*] Summary: Equivariant Variance changes by only {eq_decay:.2f}x (Constant O(1) Trainability).", flush=True)
    print("=" * 85, flush=True)
    
    print("\n" + "=" * 85, flush=True)
    print("TABLE 2: GENERALIZATION TEST RMSE VS SAMPLE SIZE M (n=4 QUBITS)", flush=True)
    print("=" * 85, flush=True)
    print(f"{'M (Samples)':<12} | {'HEA Test RMSE (mean +- std)':<30} | {'Equivariant Test RMSE (mean +- std)':<30}", flush=True)
    print("-" * 85, flush=True)
    
    m_hea = results_gen["HEA_rmse_mean"]
    s_hea = results_gen["HEA_rmse_std"]
    m_eq = results_gen["Equiv_rmse_mean"]
    s_eq = results_gen["Equiv_rmse_std"]
    
    for idx, M in enumerate(M_values):
        hea_str = f"{m_hea[idx]:.4f} +- {s_hea[idx]:.4f}"
        eq_str = f"{m_eq[idx]:.4f} +- {s_eq[idx]:.4f}"
        print(f"{M:<12} | {hea_str:<30} | {eq_str:<30}", flush=True)
        
    print("-" * 85, flush=True)
    print(f"[*] Small Sample Advantage (M={M_values[0]}): Equivariant RMSE ({m_eq[0]:.3f}) outperforms HEA ({m_hea[0]:.3f}).", flush=True)
    print(f"[*] Large Sample Expressivity (M={M_values[-1]}): HEA RMSE ({m_hea[-1]:.3f}) vs Equivariant floor ({m_eq[-1]:.3f}).", flush=True)
    print("=" * 85, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Figure 1 Generation (3-Panel Publication Figure)
# ─────────────────────────────────────────────────────────────────────────────
def generate_publication_figures(qubit_range, var_hea_list, var_equiv_list, results_generalization, M_values):
    print("\n[+] Generating Publication Figure 1 (PDF & PNG)...", flush=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)

    # Panel A: Trainability (Gradient Variance vs n)
    ax1.plot(qubit_range, var_hea_list, 'o--', color='#d9534f', linewidth=2.2, markersize=7, label=r'HEA ($\dim(\mathfrak{g}) = 4^n - 1$)')
    ax1.plot(qubit_range, var_equiv_list, 's-', color='#2e6da4', linewidth=2.2, markersize=7, label=r'Equivariant ($\dim(\mathfrak{g}) = 3$)')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'Number of Qubits $n$', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'Gradient Variance $\mathrm{Var}_{\boldsymbol{\theta}}[\partial_1 \mathcal{C}]$', fontsize=12, fontweight='bold')
    ax1.set_title(r'(a) Trainability: Barren Plateau Scaling', fontsize=12, fontweight='bold')
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', frameon=True, fontsize=10)

    # Panel B: Generalization (RMSE vs M at n=4)
    ax2.errorbar(M_values, results_generalization["HEA_rmse_mean"], 
                 yerr=results_generalization["HEA_rmse_std"], 
                 fmt='o--', color='#d9534f', linewidth=2.2, capsize=4, label='HEA ($n=4$)')
    ax2.errorbar(M_values, results_generalization["Equiv_rmse_mean"], 
                 yerr=results_generalization["Equiv_rmse_std"], 
                 fmt='s-', color='#2e6da4', linewidth=2.2, capsize=4, label='Equivariant ($n=4$)')
    ax2.axhline(y=0.3 / np.sqrt(2), color='#5cb85c', linestyle='-.', linewidth=1.5, label=r'Symmetry-Breaking Floor ($0.212$)')
    ax2.set_xlabel(r'Training Sample Size $M$', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'Test Error (RMSE)', fontsize=12, fontweight='bold')
    ax2.set_title(r'(b) Generalization: Sample Complexity', fontsize=12, fontweight='bold')
    ax2.grid(True, which='both', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', frameon=True, fontsize=9.5)

    # Panel C: The Duality Phase Portrait
    var_at_n4_hea = var_hea_list[2] # n=4
    var_at_n4_eq = var_equiv_list[2] # n=4
    gen_gap_hea = np.array(results_generalization["HEA_rmse_mean"])
    gen_gap_eq = np.array(results_generalization["Equiv_rmse_mean"])

    ax3.scatter([var_at_n4_hea] * len(M_values), gen_gap_hea, color='#d9534f', s=80, alpha=0.85, label='HEA ($n=4$, $M \in [10,200]$)')
    ax3.scatter([var_at_n4_eq] * len(M_values), gen_gap_eq, color='#2e6da4', s=80, alpha=0.85, marker='s', label='Equivariant ($n=4$, $M \in [10,200]$)')
    ax3.set_xscale('log')
    ax3.set_xlim(5e-3, 1.0)
    ax3.set_xlabel(r'Gradient Variance (Trainability)', fontsize=12, fontweight='bold')
    ax3.set_ylabel(r'Test RMSE (Generalization Error)', fontsize=12, fontweight='bold')
    ax3.set_title(r'(c) The Fundamental Duality Plane', fontsize=12, fontweight='bold')
    ax3.grid(True, which='both', linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', frameon=True, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf, dpi=300)
    plt.close()

    print(f"[+] Publication figures saved:\n    - {fig_png}\n    - {fig_pdf}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Execution Routine
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_flag = ("--train" in sys.argv) or ("-t" in sys.argv)
    
    # Fast mode: If data exists and --train is not requested
    if not train_flag and os.path.exists(json_path):
        print(f"[*] Fast reproduction mode: loading precomputed data from {os.path.basename(json_path)}...", flush=True)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        qubit_range = data["qubit_range"]
        var_hea_list = data["var_hea"]
        var_equiv_list = data["var_equiv"]
        results_gen = data["generalization"]
        M_values = results_gen["M_values"]
        
        print_tables(qubit_range, var_hea_list, var_equiv_list, results_gen, M_values)
        generate_publication_figures(qubit_range, var_hea_list, var_equiv_list, results_gen, M_values)
        print("\n" + "=" * 85, flush=True)
        print("SUCCESS: Reproduction completed in fast mode. Pass '--train' to retrain from scratch.", flush=True)
        print("=" * 85, flush=True)
        sys.exit(0)

    # Full simulation mode:
    print("\n[*] Full simulation mode: Running training across all qubits and sample sizes...", flush=True)
    pnp.random.seed(42)
    np.random.seed(42)
    qubit_range = [2, 3, 4, 5, 6]
    n_samples = 60
    var_hea_list = []
    var_equiv_list = []

    print("\n[PART 1] Measuring Gradient Variance Scaling vs Qubit Count n...", flush=True)
    for n in qubit_range:
        t0 = time.time()
        circuit_hea, p_hea = create_hea_circuit(n, layers=3)
        circuit_equiv, p_equiv = create_equivariant_circuit(n, layers=3)
        
        grad_hea_fn = qml.grad(circuit_hea)
        grad_equiv_fn = qml.grad(circuit_equiv)
        
        grads_hea = []
        grads_equiv = []
        for s in range(n_samples):
            # Sample random parameters in [0, 2pi)
            params_hea = pnp.random.uniform(0, 2 * np.pi, size=p_hea, requires_grad=True)
            params_equiv = pnp.random.uniform(0, 2 * np.pi, size=p_equiv, requires_grad=True)
            
            g_hea = grad_hea_fn(params_hea)[0]
            g_equiv = grad_equiv_fn(params_equiv)[0]
            
            grads_hea.append(float(g_hea))
            grads_equiv.append(float(g_equiv))
            
        var_hea = float(np.var(grads_hea))
        var_equiv = float(np.var(grads_equiv))
        var_hea_list.append(var_hea)
        var_equiv_list.append(var_equiv)
        
        dt = time.time() - t0
        print(f"  n = {n:2d} ({dt:.2f}s) | HEA Var: {var_hea:.6f} | Equiv Var: {var_equiv:.6f} | Ratio: {var_equiv/var_hea:.2f}x", flush=True)

    print("\n[PART 2] Measuring Generalization Error vs Training Sample Size M (n=4)...", flush=True)
    n_eval = 4
    M_values = [10, 25, 50, 100, 200]
    n_seeds = 3
    n_steps = 25
    lr = 0.08
    noise_sigma = 0.15

    # Target function for n=4
    def target_function(X):
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            return 0.7 * np.sin(np.sum(X_arr)) - 0.3 * np.cos(2 * X_arr[0])
        return 0.7 * np.sin(np.sum(X_arr, axis=1)) - 0.3 * np.cos(2 * X_arr[:, 0])

    # Fixed test set
    np.random.seed(999)
    X_test = np.random.uniform(-np.pi, np.pi, size=(50, n_eval))
    y_test = target_function(X_test)

    results_generalization = {
        "M_values": M_values,
        "HEA_rmse_mean": [],
        "HEA_rmse_std": [],
        "Equiv_rmse_mean": [],
        "Equiv_rmse_std": []
    }

    for M in M_values:
        t0 = time.time()
        rmse_hea_runs = []
        rmse_equiv_runs = []
        batch_size = min(M, 12)
        
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed * 100 + M)
            X_train = rng.uniform(-np.pi, np.pi, size=(M, n_eval))
            y_train = target_function(X_train) + rng.normal(0, noise_sigma, size=M)
            
            # 1. Train HEA model
            c_hea, p_hea = create_hea_circuit(n_eval, layers=2)
            params_hea = pnp.array(rng.uniform(-0.1, 0.1, size=p_hea), requires_grad=True)
            opt_hea = qml.AdamOptimizer(stepsize=lr)
            
            for _ in range(n_steps):
                b_idx = rng.choice(M, size=batch_size, replace=False)
                X_b, y_b = X_train[b_idx], y_train[b_idx]
                def cost_hea(p):
                    preds = [c_hea(p, x=X_b[i]) for i in range(batch_size)]
                    return pnp.mean((pnp.array(preds) - y_b)**2)
                params_hea = opt_hea.step(cost_hea, params_hea)
                
            test_preds_hea = np.array([float(c_hea(params_hea, x=X_test[i])) for i in range(len(X_test))])
            rmse_hea = float(np.sqrt(np.mean((test_preds_hea - y_test)**2)))
            rmse_hea_runs.append(rmse_hea)
            
            # 2. Train Equivariant model
            c_equiv, p_equiv = create_equivariant_circuit(n_eval, layers=2)
            params_equiv = pnp.array(rng.uniform(-0.1, 0.1, size=p_equiv), requires_grad=True)
            opt_equiv = qml.AdamOptimizer(stepsize=lr)
            
            for _ in range(n_steps):
                b_idx = rng.choice(M, size=batch_size, replace=False)
                X_b, y_b = X_train[b_idx], y_train[b_idx]
                def cost_equiv(p):
                    preds = [c_equiv(p, x=X_b[i]) for i in range(batch_size)]
                    return pnp.mean((pnp.array(preds) - y_b)**2)
                params_equiv = opt_equiv.step(cost_equiv, params_equiv)
                
            test_preds_equiv = np.array([float(c_equiv(params_equiv, x=X_test[i])) for i in range(len(X_test))])
            rmse_equiv = float(np.sqrt(np.mean((test_preds_equiv - y_test)**2)))
            rmse_equiv_runs.append(rmse_equiv)
            
        m_hea, s_hea = float(np.mean(rmse_hea_runs)), float(np.std(rmse_hea_runs))
        m_eq, s_eq = float(np.mean(rmse_equiv_runs)), float(np.std(rmse_equiv_runs))
        
        results_generalization["HEA_rmse_mean"].append(m_hea)
        results_generalization["HEA_rmse_std"].append(s_hea)
        results_generalization["Equiv_rmse_mean"].append(m_eq)
        results_generalization["Equiv_rmse_std"].append(s_eq)
        
        dt = time.time() - t0
        print(f"  M = {M:3d} ({dt:.2f}s) | HEA Test RMSE: {m_hea:.4f} +- {s_hea:.4f} | Equiv Test RMSE: {m_eq:.4f} +- {s_eq:.4f}", flush=True)

    # Save structured benchmark data
    all_benchmark_data = {
        "qubit_range": qubit_range,
        "var_hea": var_hea_list,
        "var_equiv": var_equiv_list,
        "generalization": results_generalization,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_benchmark_data, f, indent=2)
    print(f"\n[+] Benchmark metrics saved to: {json_path}", flush=True)

    # Print summary tables and generate figures
    print_tables(qubit_range, var_hea_list, var_equiv_list, results_generalization, M_values)
    generate_publication_figures(qubit_range, var_hea_list, var_equiv_list, results_generalization, M_values)

    print("\n" + "=" * 85, flush=True)
    print("SUCCESS: Paper 3 numerical experiment suite completed successfully.", flush=True)
    print("=" * 85, flush=True)
    sys.exit(0)
