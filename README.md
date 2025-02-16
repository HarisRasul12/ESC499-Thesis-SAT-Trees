# SATreeCraft: SAT-Based Decision Tree & Clustering Solver

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/satreecraft.svg)](https://pypi.org/project/satreecraft/)
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

### Classification Example

```python
import numpy as np
from satree import SATreeCraft, SATreeClassifier

# Assume X is your feature matrix and y are the true labels
X = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [6.7, 3.1, 4.7, 1.5],
    [5.6, 3.0, 4.1, 1.3],
    [6.3, 3.3, 6.0, 2.5]
])
y = np.array([0, 0, 1, 1, 2])
features = np.array(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
# Define which features are categorical (if any) and which are numerical
features_categorical = []  # e.g., if none are categorical
features_numerical = features  # all features are numerical here

# For a minimum-height tree classification problem:
solver = SATreeCraft(
    dataset=X,
    features=features,
    labels=np.unique(y),
    true_labels_for_points=y,
    features_categorical=features_categorical,
    features_numerical=features_numerical,
    classification_objective='min_height',  # or 'max_accuracy'
    fixed_depth=None,  # Not used for min_height objective
    tree_structure='Complete'
)
solver.solve()

# Build a classifier from the obtained decision tree model
classifier = SATreeClassifier(solver.model)
accuracy = classifier.score(X, y)
print("Training accuracy:", accuracy)
```

### Clustering Example

```python
# For a clustering problem, define additional parameters:
k_clusters = 3
depth = 3  # fixed depth for clustering trees
epsilon = 0.5
# Optionally define must-link and cannot-link pairs (as numpy arrays)
ml_pairs = np.array([[0, 1]])
cl_pairs = np.array([[2, 3]])

# Initialize SATreeCraft for clustering (set is_classification=False)
cluster_solver = SATreeCraft(
    dataset=X,
    features=features,
    labels=np.unique(y),
    true_labels_for_points=y,
    features_categorical=features_categorical,
    features_numerical=features_numerical,
    classification_objective='min_height',  # not used in clustering mode
    fixed_depth=depth,
    k_clusters=k_clusters,
    clustering_objective='max_diameter',  # or 'bicriteria'
    is_clustering=True,
    epsilon=epsilon,
    cl_pairs=cl_pairs,
    ml_pairs=ml_pairs
)
cluster_solver.solve()

# The clustering solution is available as:
print("Cluster assignments:", cluster_solver.cluster_assignments)
print("Cluster diameters:", cluster_solver.cluster_diameters)
```

### Loandra Support

If you prefer to use the external Loandra solver for potentially faster or alternative MaxSAT solving, simply call:

```python
loandra_path = "/path/to/loandra"  # absolute path to your Loandra installation
solver.solve_loandra(loandra_path)
```
The solver will export the CNF file in DIMACS format, call Loandra, and postprocess the solution.


---

## License
SATreeCraft is fully open source. This project is licensed under the MIT License. 
See the [LICENSE](LICENSE.txt) file for details.

---

## Contact

For questions, feedback, or support, please open an issue on GitHub or contact the development team at [your.email@example.com](mailto:your.email@example.com).

---

*SATreeCraft is developed and maintained by [Your Name or Organization].*
