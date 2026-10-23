# Numerical Reproducibility: A Fundamental Duality in Quantum Machine Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.30+-orange.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the self-contained numerical simulation scripts and benchmark datasets supporting the paper:

> **A Fundamental Duality in Quantum Machine Learning: Trainability vs. Generalization via Dynamical Lie Algebras**  
> **Author:** Ayad Alhusseiny (*College of Science, University of Kerbala*)  
> **Correspondence:** `ayad.a@uokerbala.edu.iq`

---

## Overview

Dynamical Lie Algebra (DLA) dimension, rather than Euclidean parameter count, controls the fundamental trade-off between optimization trainability and statistical generalization in Quantum Machine Learning (QML).

For continuous Parameterized Quantum Circuits (PQCs) on $n$ qubits with dynamical Lie algebra $\mathfrak{g}_n = \mathrm{Lie}_{\mathbb{R}}\{-i H_1, \dots, -i H_L\} \subseteq \mathfrak{su}(2^n)$ and dynamical Lie group $\mathcal{G}_n = e^{\mathfrak{g}_n}$:

1. **Optimization Trainability (Lemma 1):** Under joint state-observable non-degeneracy and normalized Casimir-Killing coefficient $C_O = \Theta(1)$, the Haar-averaged gradient variance scales inversely with the DLA dimension:
   $$\overline{\mathrm{Var}}_{\boldsymbol{\theta}}[\partial_k \mathcal{C}] = \Theta\!\left( \frac{1}{\dim(\mathfrak{g}_n)} \right).$$

2. **Minimax Generalization (Lemma 2):** Under local Riemannian packing over the adjoint orbit $\mathcal{O}_O = \{U^\dagger O U : U \in \mathcal{G}_n\}$, the minimax sample complexity required to achieve risk $\varepsilon$ scales directly with the orbit dimension:
   $$M^*(\varepsilon) = \Theta\!\left( \frac{\dim(\mathcal{O}_O)}{\varepsilon^2} \right),$$
   where $\dim(\mathcal{O}_O) = \dim(\mathfrak{g}_n) - \dim(G_O) \le \dim(\mathfrak{g}_n)$.

3. **The Universal Duality Invariant (Theorem 1):** Their depth-normalized product satisfies:
   $$\overline{\mathrm{Var}}_{\boldsymbol{\theta}}[\nabla \mathcal{C}] \cdot M^*(\varepsilon) = \Theta\!\left( \frac{\dim(\mathcal{O}_O)}{\dim(\mathfrak{g}_n) \, \varepsilon^2} \right),$$
   which collapses to the universal, dimension-free invariant $\Theta(1/\varepsilon^2)$ for all generic observables where $\dim(\mathcal{O}_O) = \Theta(\dim(\mathfrak{g}_n))$.

| Architecture Family | DLA Dimension $\dim(\mathfrak{g}_n)$ | Gradient Variance (Trainability) | Sample Complexity (Generalization) | Barren Plateaus? |
| :--- | :---: | :---: | :---: | :---: |
| **Hardware-Efficient (HEA)** | $\mathcal{O}(4^n)$ | $\mathcal{O}(2^{-n})$ (Vanishing) | $\mathcal{O}(4^n / \varepsilon^2)$ (Exponential) | **Yes** (Severe) |
| **Permutation-Equivariant** | $\mathcal{O}(1)$ | $\Omega(1)$ (Constant) | $\mathcal{O}(1 / \varepsilon^2)$ (Sample-Efficient) | **No** (Trainable) |

---

## Repository Contents

* [`experiments_and_figures.py`](experiments_and_figures.py): Standalone, vectorized PennyLane simulation script that simulates both Hardware-Efficient and Permutation-Equivariant circuits across $n \in \{2, 3, 4, 5, 6\}$ qubits, prints Table 1 and Table 2 to the console, and generates Figure 1 (`fig_duality_experiments.pdf` and `.png`).
* [`experiments_results.json`](experiments_results.json): Precomputed raw benchmark evaluation data across all independent random seeds with exact means and standard deviations.
* [`fig_duality_experiments.pdf`](fig_duality_experiments.pdf): Vector publication graphic of Figure 1.
* [`fig_duality_experiments.png`](fig_duality_experiments.png): High-resolution raster preview of Figure 1.
* [`requirements.txt`](requirements.txt): Minimal Python package dependencies.
* [`LICENSE`](LICENSE): MIT License (Copyright 2026 Ayad Alhusseiny).

---

## Installation & Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ayadalhusseiny/qml-dla-trainability-generalization.git
cd qml-dla-trainability-generalization
```

### 2. Set up a Python environment
Python 3.9 or higher is recommended:
```bash
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Reproducing Empirical Results & Figures

### Fast Reproduction (~3 seconds)
To immediately display Table 1, Table 2, and regenerate Figure 1 from precomputed benchmark data:
```bash
python experiments_and_figures.py
```

### Full Simulation from Scratch
To re-run all variational PennyLane simulations across all qubit counts ($n=2, 3, 4, 5, 6$) and sample sizes ($M \in \{10, 25, 50, 100, 200\}$) over 3 independent random seeds:
```bash
python experiments_and_figures.py --train
```

---

## Key Experimental Results

### Table 1: Gradient Variance Scaling (Trainability)
```text
=====================================================================================
TABLE 1: GRADIENT VARIANCE SCALING (TRAINABILITY & BARREN PLATEAU SCALING)
=====================================================================================
n    | HEA dim(g)   | HEA Var[dC]    | Equiv dim(g) | Equiv Var[dC]  | Variance Ratio
-------------------------------------------------------------------------------------
2    | 15           | 0.08148        | 3            | 0.35591        | 4.37x         
3    | 63           | 0.04288        | 3            | 0.38630        | 9.01x         
4    | 255          | 0.03374        | 3            | 0.35876        | 10.63x        
5    | 1023         | 0.01212        | 3            | 0.31908        | 26.33x        
6    | 4095         | 0.00473        | 3            | 0.36513        | 77.18x        
-------------------------------------------------------------------------------------
[*] Summary: HEA Variance decays by 17.22x from n=2 to n=6 (Exponential Barren Plateau).
[*] Summary: Equivariant Variance changes by only 1.03x (Constant O(1) Trainability).
=====================================================================================
```

### Table 2: Generalization Test RMSE vs Training Sample Size $M$ ($n=4$)
```text
=====================================================================================
TABLE 2: GENERALIZATION TEST RMSE VS SAMPLE SIZE M (n=4 QUBITS)
=====================================================================================
M (Samples)  | HEA Test RMSE (mean +- std)    | Equivariant Test RMSE (mean +- std)          
-------------------------------------------------------------------------------------
10           | 0.6130 +- 0.0084               | 0.3020 +- 0.0280             
25           | 0.5246 +- 0.0254               | 0.2681 +- 0.0024             
50           | 0.5099 +- 0.0071               | 0.2919 +- 0.0312             
100          | 0.5091 +- 0.0330               | 0.2949 +- 0.0131             
200          | 0.5035 +- 0.0317               | 0.2951 +- 0.0043             
-------------------------------------------------------------------------------------
[*] Small Sample Advantage (M=10): Equivariant RMSE (0.302) outperforms HEA (0.613).
[*] Large Sample Expressivity (M=200): HEA RMSE (0.503) vs Equivariant floor (0.295).
=====================================================================================
```

---

## Citation

If you find this repository or numerical benchmarks useful in your research, please cite:

```bibtex
@article{alhusseiny2026duality,
  title={A Fundamental Duality in Quantum Machine Learning: Trainability vs. Generalization via Dynamical Lie Algebras},
  author={Alhusseiny, Ayad},
  journal={Physical Review Letters (under review)},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
