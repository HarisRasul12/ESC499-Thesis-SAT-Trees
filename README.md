# SATreeCraft: SAT-Based Decision Tree & Clustering Solver

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.8%20to%203.12-blue.svg"></a>

**Version:** v1.2 – January 2025

---

## Overview

SATreeCraft is a Python library that implements state-of-the-art, SAT-based exact optimization techniques for constructing interpretable decision trees and solving clustering problems. By transforming both classification and clustering tasks into satisfiability (SAT) formulations, SATreeCraft guarantees:

- **Optimal tree construction:** Find the minimal-height decision tree that fits your training data with 100% accuracy or optimize for maximum accuracy given a fixed depth.
- **Flexible feature support:** Handle both numerical and categorical features seamlessly.
- **Clustering objectives:** Beyond decision trees, SATreeCraft includes dedicated modules for clustering, enabling objectives such as minimizing the maximum cluster diameter or a bicriteria optimization approach.
- **Solver Integration:** Built-in support for both standard SAT/MaxSAT solvers and external integration with the Loandra solver via DIMACS file export.
- **User-Friendly Interface:** Comes with an intuitive set of classes (e.g., `SATreeClassifier`) and Jupyter Notebook demos to help users quickly get started.

---

## Key Features

- **Exact Optimization via SAT:** Formulate the decision tree and clustering problems as SAT/CNF (or weighted CNF) instances to guarantee an optimal solution under the given constraints.
- **Multiple Objectives for Classification:**
    - *Minimum Height Trees:* Find the smallest tree that achieves 100% training accuracy.
    - *Fixed Depth Maximum Accuracy:* Optimize classification performance given a user-specified tree depth.
- **Clustering Capabilities:**
    - Support for clustering formulations with objectives such as maximizing minimum intra-cluster separation (max diameter) or integrating bicriteria constraints.
- **Solver Flexibility:**
    - Use built-in SAT solvers via PySAT.
    - Optionally, leverage the external Loandra MaxSAT solver for enhanced performance.
- **Comprehensive Documentation & Demos:**
    - Detailed Jupyter Notebook demos for classification, clustering, and Loandra integration.
    - Export CNF/DIMACS files for external solver analysis.
- **Modular and Extensible Design:** Easily integrate new constraints, modify objective functions, or extend the solver interface.

---

## Installation

SATreeCraft supports Python 3.8 and above. You can install it via pip:

```bash
git clone https://github.com/yourusername/satreecraft.git
cd satreecraft
pip install -e .
```

## Dependencies

SATreeCraft is developed for Python 3.8+ and uses a [pyproject.toml](./pyproject.toml) file for dependency management. The core dependencies include:

- **NumPy (>=1.20)**
- **SciPy (>=1.6)**
- **scikit-learn (>=0.24)**
- **python-sat (>=0.1.7.dev6)**
  **Note:** Requires a POSIX-compliant OS, GNU make, patch, and a C/C++ compiler with C++11 support.  
  Recommended installation with extras:
  ```bash
  pip install 'python-sat[aiger,approxmc,cryptosat,pblib]'
  ```
See [python-sat installation details](https://pysathq.github.io/installation/).

- **Graphviz (>=0.13)**
- **matplotlib (>=3.3)**
- **pandas (>=1.0)**

### Optional External Solver Support: LOANDRA

SATreeCraft can integrate with external SAT solvers, such as the Loandra MaxSAT solver, for alternative solving methods.

- **Loandra (MaxSAT Solver)**:
    - Provides an alternative approach for processing DIMACS files through SATreeCraft.
    - **Requirement:** Supply the path to the Loandra executable. SATreeCraft will generate DIMACS files and parse Loandra’s output automatically.
    - **Reference:**  
      Berg, J., Demirović, E. and Stuckey, P.J., 2019. *Core-boosted linear search for incomplete MaxSAT.* In *Integration of Constraint Programming, Artificial Intelligence, and Operations Research: 16th International Conference, CPAIOR 2019, Thessaloniki, Greece, June 4–7, 2019, Proceedings 16* (pp. 39-56). Springer International Publishing.

For more details, visit the [Loandra GitHub repository](https://github.com/jezberg/loandra).
Users must set the path to the Loandra solver in accordance with their solving method. For more details, visit the [Loandra GitHub repository](https://github.com/jezberg/loandra).

---

## Quick Start

Below is a brief example demonstrating how to construct a decision tree classifier from your data and evaluate its performance.

### Fixed Height Tree Example

```python

from satree.utils import TreeDataLoaderBinaryNumerical
from satree.SATreeCraft import SATreeCraft # Tree solver framework

file_path_to_test = '../data/wine/wine.data'
delimiter = ','
label_position = 0

data_loader = TreeDataLoaderBinaryNumerical(file_path=file_path_to_test, delimiter=delimiter, label_position= label_position)

max_accuracy_numerical_problem = SATreeCraft(dataset=data_loader.dataset,
                                             features=data_loader.features,labels=data_loader.labels,
                                             true_labels_for_points=data_loader.true_labels_for_points,
                                             classification_objective='max_accuracy',
                                             fixed_depth=2)

max_accuracy_numerical_problem.solve()

# # Or, use Loandra for faster solving
# # Path to the loandra executable (recommended to use for faster solving). To install, follow: https://github.com/jezberg/loandra
# loandra_path = "/.../loandra"  # Path to your loandra executable (Change as needed)
# max_accuracy_numerical_problem.solve_loandra(loandra_path= loandra_path)

print("Final Model: ", max_accuracy_numerical_problem.model)
print("Min cost found: ", max_accuracy_numerical_problem.min_cost)
```

### Clustering Example

```python
import numpy as np

from satree.utils import TreeDataLoaderBinaryNumerical # Dataloader and K-fold mechanism
from satree.SATreeCraft import SATreeCraft # Tree solver framework

file_path_to_test = '../data/wine/wine.data'
delimiter = ','
label_position = 0

data_loader = TreeDataLoaderBinaryNumerical(file_path=file_path_to_test, delimiter=delimiter, label_position= label_position)

epsilon, k_clusters, depth = 0.1, 3, 3

cl_pairs = np.array([]) # cannot-link pairs (add as needed)
ml_pairs = np.array([]) # must-link pairs (add as needed)

clustering_problem = SATreeCraft(dataset=data_loader.dataset,
                                 features=data_loader.features,
                                 k_clusters=k_clusters,
                                 ml_pairs=ml_pairs,
                                 cl_pairs=cl_pairs,
                                 epsilon=epsilon,
                                 fixed_depth=depth)

clustering_problem.solve() 

# # Or, use Loandra for faster solving
# # Path to the loandra executable (recommended to use for faster solving). To install, follow: https://github.com/jezberg/loandra
# loandra_path = "/.../loandra"  # Path to your loandra executable (Change as needed)
# clustering_problem.solve_loandra(loandra_path)

print(f"Results for depth {depth}:")
print(clustering_problem.cluster_assignments)
print(clustering_problem.cluster_diameters)
```

### Loandra Support

If you prefer to use the external Loandra solver for potentially faster or alternative MaxSAT solving, simply call:

```python
loandra_path = "/path/to/loandra"  # absolute path to your Loandra installation
solver.solve_loandra(loandra_path)
```
The solver will export the CNF file in DIMACS format, call Loandra, and postprocess the solution.

---

## SATree Test Suite

This folder contains a complete test suite for the SATree Python library. The tests are written using pytest,
cover unit tests for core classes (such as `SATreeClassifier` and `SATreeCraft`), submodules (classification, clustering,
loandra support, treemodder, and utils), and include integration tests with fixtures and mocking.

### Running the Tests

To run all tests:

```bash
pytest
```

---

## License
SATreeCraft is fully open source. This project is licensed under the MIT License. 
See the [LICENSE](LICENSE.txt) file for details.
