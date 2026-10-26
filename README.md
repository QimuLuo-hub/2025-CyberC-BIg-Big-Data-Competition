# Gold Cluster Energy Prediction with Graph Neural Networks

**2025 IEEE CyberC Big-Big Data Analytics Competition** · Research code and numerical experiments

This project studies the prediction and interpretation of **Au20 gold-cluster energies**. We combine physically motivated baseline potentials with machine learning to improve energy estimates, analyze stable atomic configurations, and test model behavior under structural perturbations.

## Research question

Accurate cluster-energy estimates are useful for identifying stable atomic structures, but classical interatomic potentials can be inaccurate for small clusters with complex many-body interactions. Can a geometry-aware model learn corrections to classical estimates while producing predictions that remain meaningful when atomic positions change?

## Methods

Our approach uses **delta learning**: a classical energy estimate is aligned with the reference values, and a learned model predicts the remaining residual. We compare classical corrections, multilayer perceptrons (MLPs), and a **message-passing neural network (MPNN)**. The graph model represents atomic neighborhoods and uses geometric information to capture interactions that are difficult to describe using only global summary features.

The project comprises three connected investigations:

1. **Energy prediction:** compare classical baselines, MLP-based residual learning, and graph-based residual learning using prediction errors.
2. **Low-energy structure analysis:** examine the energy distribution and the geometric and graph-topological properties of low-energy configurations, including bond lengths, coordination, and connectivity.
3. **Sensitivity analysis:** perturb atomic coordinates and examine how predictions and residuals respond, to assess numerical smoothness and robustness.

## Key findings

- The MPNN obtained **MAE 0.366**, compared with **0.977** for the linear baseline (approximately **62% lower MAE**) in the reported comparison.
- The lowest-energy configuration analyzed has a **tetrahedral structure**. Geometry and network descriptors help explain common patterns across other low-energy isomers.
- Perturbation experiments indicate that the graph-based predictions remain smooth and bounded under the tested coordinate changes. The full report gives the perturbation settings and evaluation details.

## Repository guide

The main experimental scripts are in [`CyberC_Final_Code/`](CyberC_Final_Code/):

| File | Purpose |
| --- | --- |
| [`Task1_classical.py`](CyberC_Final_Code/Task1_classical.py) | Classical energy baselines and alignment. |
| [`Task1_hybrid_delta_learning_MLP.py`](CyberC_Final_Code/Task1_hybrid_delta_learning_MLP.py) | MLP-based residual-learning comparison. |
| [`Task1_gnn_weighted.py`](CyberC_Final_Code/Task1_gnn_weighted.py) | Graph-based energy prediction using an MPNN. |
| [`Task2-energydistribution_graph.py`](CyberC_Final_Code/Task2-energydistribution_graph.py) | Energy-distribution and structure-related visualization. |
| [`Task3_sensitivity_with_corrds.py`](CyberC_Final_Code/Task3_sensitivity_with_corrds.py) | Coordinate-perturbation and sensitivity experiments. |
| [`out/`](CyberC_Final_Code/out/) | Saved evaluation outputs, including model residuals. |

The repository also contains a [`Plot/`](Plot/) directory with figure-generation materials. The files above are starting points for navigating the work; they are research scripts rather than a single packaged training pipeline.

## Reproducibility notes

The code uses Python and machine-learning/scientific-computing libraries. Some scripts refer to local input data, saved model outputs, or paths specific to the original experiments. **Check the arguments and paths in each script before running it.** This repository does not currently provide a verified one-command reproduction procedure.

## Project context

This work was prepared for the **IEEE CyberC 2025 Big-Big Data Analytics Competition**. It illustrates an interest in building ML methods that are useful in scientific settings, while examining not only predictive accuracy but also structural interpretation and stability.
