# Quantum Entanglement as Super-Confounding: From Bell's Theorem to Robust Machine Learning

This repository contains the official source code and data for the paper, "Quantum Entanglement as Super-Confounding: From Bell's Theorem to Robust Machine Learning" ([paper at arxiv](https://arxiv.org/abs/2508.19327)).

## Abstract

Bell's theorem reveals a profound conflict between quantum mechanics and local realism, a conflict we reinterpret through the modern lens of causal inference. We propose and computationally validate a framework where quantum entanglement acts as a "super-confounding" resource, generating correlations that violate the classical causal bounds set by Bell's inequalities. This work makes three key contributions: First, we establish a physical hierarchy of confounding (Quantum > Classical) and introduce Confounding Strength (CS) to quantify this effect. Second, we provide a circuit-based implementation of the quantum DO-calculus to distinguish causality from spurious correlation. Finally, we apply this calculus to a quantum machine learning problem, where causal feature selection yields a statistically significant 11.3% average absolute improvement in model robustness. Our framework bridges quantum foundations and causal AI, offering a new, practical perspective on quantum correlations.


## Repository Structure

* **/src**: Contains the core Python scripts for running the computational experiments to generate the raw data.
* **/figures_and_data**: Contains a subdirectory for each experiment's analysis. Each subdirectory holds the Python script to generate the corresponding figure and the necessary data files (JSON logs).
* **/figures**: Contains the final, generated figures (PDF format) as they appear in the manuscript.

## Mapping Experiments to Source Files

**Important Note:** The experiment numbering in the manuscript differs from the internal file numbering used in this repository. The following table provides a clear mapping:

| Paper Experiment | Data Generation Script (`/src`) | Description |
| :--- | :--- | :--- |
| **Experiment 1** | `ex0_framework_validation.py` | Validates the foundational assumptions of the framework. |
| **Experiment 2** | `ex1new_confounding_hierarchy.py` | Demonstrates the confounding hierarchy (Fig. 2). |
| **Experiment 3** | `ex2new_confounding_quantification.py`| Quantifies the link between entanglement and CS (Fig. 3). |
| **Experiment 4** | `ex3new_quantum_do_calculus.py` | Implements the quantum $\mathcal{DO}$-calculus (Fig. 4). |
| **Experiment 5** | `ex7_multi_seed.py` | Demonstrates causal feature selection in QML (Fig. 5). |
| **Hardware Exp.** | `ex1a_no_confounding_ionq.py` | "No Confounding" scenario run on IonQ hardware. |
| **Hardware Exp.** | `ex1b_quantum_confounding_ionq.py` | "Quantum Confounding" scenario run on IonQ hardware. |


## Reproducibility Guide

### 1. Installation

To set up the necessary environment, please install the required packages from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 2. Workflow

This repository is structured to separate data generation from analysis and plotting.

**Step A: Using Pre-generated Data (Recommended)**

The `/figures_and_data` directory already contains all the raw data files (JSON logs) needed to reproduce the figures in the paper. You can proceed directly to Step C.

**Step B: Regenerating the Data (Optional)**

If you wish to regenerate the raw data, run the experiment scripts located in the `/src` directory. The generated JSON files should be placed in the corresponding subdirectory within `/figures_and_data` for the plotting scripts to find them.

```bash
# Example for Experiment 2 data
python src/ex1new_confounding_hierarchy.py
# Move the resulting .json file to figures_and_data/Experiment_2/
```

**Step C: Generating the Figures**

If you wish to regenerate the raw data, run the experiment scripts located in the `/src` directory. The generated JSON files should be placed in the corresponding subdirectory within `/figures_and_data` for the plotting scripts to find them.

```bash
# Example for Figure 2 (associated with Experiment 2)
python figures_and_data/Experiment_2/fig_2_CS_confound_hierarchy.py
```

## License

This project is licensed under the MIT License.
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


### Citation
@misc{kang2025quantumentanglementsuperconfoundingbells,
      title={Quantum Entanglement as Super-Confounding: From Bell's Theorem to Robust Machine Learning},
      author={Pilsung Kang},
      year={2025},
      eprint={2508.19327},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2508.19327},
}
