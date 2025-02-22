"""
=========== Module Description ===========

This module implements a SAT-based framework for constructing fixed-depth decision trees
tailored to categorical classification problems. It encodes the tree structure as a weighted
CNF (WCNF) formula by translating the decision tree constraints into SAT clauses. The encoding
combines hard constraints—ensuring a valid binary tree structure, proper feature selection,
and consistent label assignment at the leaves—with soft constraints that favor solutions
maximizing classification accuracy.

Key mathematical and algorithmic steps include:
  - Building a complete binary tree of a specified depth and generating corresponding SAT
    literals. These literals represent:
      • Decision splits at branching nodes (‘a’ literals),
      • Data point routing through the tree (‘s’ literals),
      • Assignment of data points to leaves (‘z’ literals),
      • Label assignments at the leaves (‘g’ literals).
  - Constructing a set of SAT clauses specific to fixed-height categorical problems. This
    includes clauses for feature selection, ordering constraints (using both categorical
    grouping and numerical sorting), and classification consistency.
  - Solving the resulting SAT/Partial MaxSAT formulation to obtain a model that satisfies
    the constraints (or minimizes the cost in a weighted setting), thereby yielding a decision
    tree that best fits the training data.
  - Post-processing the SAT solution to compute and assign thresholds (decision boundaries)
    for the branching nodes, and visualizing the finalized decision tree.

Overall, the module provides functions to build the complete SAT encoding, solve for an optimal
tree structure, and extract a decision tree that is both structurally valid and optimized for
classification accuracy.
"""

from typing import List, Dict, Any

import numpy as np
from pysat.formula import WCNF

from satree.classification.sat_clauses import add_clauses_for_features_and_paths, add_classification_clauses
from satree.common_sat_clauses import add_feature_selection_clauses_for_branching_nodes


def build_clauses_categorical_fixed(literals: Dict[str, int],
                                    dataset: np.ndarray,
                                    branch_nodes: List[int],
                                    leaf_nodes: List[int],
                                    num_features: int,
                                    features_categorical: List[str],
                                    features_numerical: List[str],
                                    labels: List[Any],
                                    true_labels: List[Any]) -> WCNF:
    """
    Generates SAT clauses for fixed-depth decision trees that handle both categorical and numerical features.

    This function extends the core encoding by incorporating constraints specific to categorical features—
    including grouping indices by category—to support direct branching without binary expansion. It aligns with
    the power set branching extension discussed in Section 4 of the paper.

    Args:
        literals: A dictionary mapping literals to variable indices.
        dataset: The dataset, a list of tuples representing data points.
        branch_nodes: Indices of branching nodes.
        leaf_nodes: Indices of leaf nodes.
        num_features: Number of features in the dataset.
        features_categorical: Indices of categorical features.
        features_numerical: Indices of numerical features.
        labels: Possible class labels for the data points.
        true_labels: True labels for the data points

    Returns:
        A WCNF object containing all the clauses with unit weighs for cost
    """
    wcnf = WCNF()
    wcnf = add_feature_selection_clauses_for_branching_nodes(wcnf, literals, branch_nodes, num_features)
    wcnf = add_clauses_for_features_and_paths(wcnf, literals, dataset, branch_nodes, leaf_nodes, num_features,
                                              features_categorical, features_numerical, labels)
    wcnf = add_classification_clauses(wcnf, literals, dataset, leaf_nodes, true_labels)

    return wcnf
