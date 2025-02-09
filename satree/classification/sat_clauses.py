"""
=========== Module Description ===========

This module provides a suite of helper functions for constructing SAT (and MaxSAT) clauses
to encode decision tree based classification problems. It includes utilities to:
  - Append direction clauses for branching nodes.
  - Add data point direction and path validity clauses.
  - Construct feature selection clauses and combine them with data point constraints.
  - Build maxSAT clauses that mix hard constraints (e.g., tree structure integrity)
    with soft constraints (e.g., maximizing correct classifications).
  - Incorporate redundant constraints and path deviation clauses to prune the search space.

These functions are designed to work with both CNF and WCNF representations (from the pysat library)
and support the encoding of trees that handle both numerical and categorical features.
"""

from typing import List, Dict, Any, Union

import numpy as np
from pysat.formula import CNF, WCNF

from satree.classification.core import compute_ordering_with_categorical
from satree.common_sat_clauses import construct_feature_selection_clauses, append_direction_clauses, \
    add_path_validity_and_deviation_clauses


def construct_maxsat_clauses(wcnf: Union[WCNF, CNF],  # Use WCNF or CNF as appropriate
                             literals: Dict[str, int],
                             dataset: np.ndarray,
                             branch_nodes: List[int],
                             leaf_nodes: List[int],
                             num_features: int,
                             labels: List[Any]) -> WCNF:
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding.

    Args:
        wcnf: The WCNF object to which the clauses will be added.
        literals: A dictionary mapping literals to variable indices.
        dataset: The dataset, a list of tuples representing data points.
        branch_nodes: Indices of branching nodes.
        leaf_nodes: Indices of leaf nodes.
        num_features: Number of features in the dataset.
        labels: Possible class labels for the data points.

    Returns:
        A WCNF object containing all the clauses, with hard clauses for the tree structure and soft clauses for maximizing correctly classified points.
    """

    wcnf = construct_feature_selection_clauses(wcnf, literals, dataset, branch_nodes, leaf_nodes, num_features)

    # Clause (8): Each leaf node is assigned at most one label
    for t in leaf_nodes:
        for c in range(len(labels)):
            for cp in range(c + 1, len(labels)):
                wcnf.append([-literals[f'g_{t}_{labels[c]}'], -literals[f'g_{t}_{labels[cp]}']])

    return wcnf


def add_classification_clauses(wcnf: WCNF,
                               literals: Dict[str, int],
                               dataset: np.ndarray,
                               leaf_nodes: List[int],
                               true_labels: List[Any]) -> WCNF:
    """
    Adds classification clauses to the WCNF object.

    This function adds hard clauses to ensure that a data point ends up in a leaf node with the correct label,
    and soft clauses to maximize the number of correctly classified data points.

    Args:
        wcnf: The WCNF object to which the clauses will be added.
        literals: A dictionary mapping literals to variable indices.
        dataset: The dataset, a list of tuples representing data points.
        leaf_nodes: Indices of leaf nodes.
        true_labels: The true labels for the data points.

    Returns:
        The updated WCNF object with the added classification clauses.
    """
    # New Hard Clause (12) for ensuring pi is true only when xi ends up in a leaf node with the correct label, REMOVED (CLAUSE 11)
    for i, xi in enumerate(dataset):
        for t in leaf_nodes:
            label = true_labels[i]
            # This adds the clause (¬pi ∨ ¬zi,t ∨ gt,γ(xi))
            wcnf.append([-literals[f'p_{i}'], -literals[f'z_{i}_{t}'], literals[f'g_{t}_{label}']])

    # Add the soft clauses (13) for each data point being correctly classified
    for i in range(len(dataset)):
        wcnf.append([literals[f'p_{i}']], weight=1)

    return wcnf


def add_clauses_for_features_and_paths(cnf: Union[WCNF, CNF],  # Use WCNF or CNF as appropriate
                                       literals: Dict[str, int],
                                       dataset: np.ndarray,
                                       branch_nodes: List[int],
                                       leaf_nodes: List[int],
                                       num_features: int,
                                       features_categorical: List[str],
                                       features_numerical: List[str],
                                       labels: List[Any]) -> Union[WCNF, CNF]:
    """
    Adds clauses for feature selection, path validity, and label assignment to the CNF object.

    This function adds clauses to ensure proper feature selection, path validity from right and left traversal,
    deviations for data points not ending in leaf nodes, and label assignment for leaf nodes.

    Args:
        cnf: The CNF object to which the clauses will be added.
        literals: A dictionary mapping literals to variable indices.
        dataset: The dataset, a list of tuples representing data points.
        branch_nodes: Indices of branching nodes.
        leaf_nodes: Indices of leaf nodes.
        num_features: Number of features in the dataset.
        features_categorical: List of categorical feature indices.
        features_numerical: List of numerical feature indices.
        labels: Possible class labels for the data points.

    Returns:
        The updated CNF object with the added clauses.
    """
    # Clauses (16), (17), and (18)
    for j in range(num_features):
        ordering = compute_ordering_with_categorical(dataset, j, features_categorical)
        for t in branch_nodes:
            for i in range(len(ordering) - 1):
                i_index, ip_index = ordering[i], ordering[i + 1]
                if str(j) in features_categorical:
                    # Clause (18) and (17) for categorical features
                    if dataset[i_index, j] == dataset[ip_index, j]:
                        cnf.append(
                            [-literals[f'a_{t}_{j}'], -literals[f's_{i_index}_{t}'], literals[f's_{ip_index}_{t}']])
                        cnf.append(
                            [-literals[f'a_{t}_{j}'], literals[f's_{i_index}_{t}'], -literals[f's_{ip_index}_{t}']])
                else:
                    # Clause (16) and (17) for numerical features
                    if float(dataset[i_index, j]) < float(dataset[ip_index, j]):
                        append_direction_clauses(cnf, literals, t, j, i_index, ip_index, append_both=False)
                    if float(dataset[i_index, j]) == float(dataset[ip_index, j]):
                        append_direction_clauses(cnf, literals, t, j, i_index, ip_index, append_both=True)

    cnf = add_path_validity_and_deviation_clauses(cnf, literals, dataset, leaf_nodes)

    # Clause (22): Each leaf node is assigned at most one label
    for t in leaf_nodes:
        for c in range(len(labels)):
            for cp in range(c + 1, len(labels)):
                cnf.append([-literals[f'g_{t}_{labels[c]}'], -literals[f'g_{t}_{labels[cp]}']])

    # Clause (23) and (24)
    for t in branch_nodes:
        for j in range(num_features):
            ordering = compute_ordering_with_categorical(dataset, j, features_categorical)
            if str(j) in features_categorical or str(j) in features_numerical:
                cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{ordering[0]}_{t}']])
            if str(j) in features_numerical:
                cnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{ordering[-1]}_{t}']])

    return cnf


