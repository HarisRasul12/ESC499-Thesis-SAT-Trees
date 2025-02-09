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

from satree.classification.classification_core import compute_ordering_with_categorical, get_ancestors, compute_ordering


def append_direction_clauses(cnf: Union[WCNF, CNF],  # Use WCNF or CNF as appropriate
                             literals: Dict[str, int],
                             t: int,
                             j: int,
                             i_index: int,
                             ip_index: int,
                             append_both: bool = False) -> None:
    """
    Appends direction clauses for a given branching node and feature.

    This function appends:
      - Clause 1: [-a_{t}_{j}, s_{i_index}_{t}, -s_{ip_index}_{t}]
      - Clause 2: [-a_{t}_{j}, -s_{i_index}_{t}, s_{ip_index}_{t}]
        (this second clause is appended only if append_both is True)

    Args:
        cnf: The CNF (list of clauses) to which the new clauses will be appended.
        literals: A dictionary mapping literal names (as strings) to their variable indices.
        t: The index of the current branching node.
        j: The feature index.
        i_index: The index of the left data point.
        ip_index: The index of the right data point.
        append_both: If True, both clauses will be appended; if False, only the first clause is appended.
    """
    cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i_index}_{t}'], -literals[f's_{ip_index}_{t}']])
    if append_both:
        cnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{i_index}_{t}'], literals[f's_{ip_index}_{t}']])


def add_data_point_clauses(cnf: CNF,
                           literals: Dict[str, int],
                           dataset: np.ndarray,
                           branch_nodes: List[int],
                           leaf_nodes: List[int],
                           num_features: int) -> CNF:
    """
    Adds data point direction and path validity clauses to the CNF object.

    This function adds clauses to ensure proper data point direction based on feature values,
    path validity from right and left traversal, and deviations for data points not ending in leaf nodes.

    Args:
        cnf: The CNF object to which the clauses will be added.
        literals: A dictionary mapping literals to variable indices.
        dataset: The dataset, a list of tuples representing data points.
        branch_nodes: Indices of branching nodes.
        leaf_nodes: Indices of leaf nodes.
        num_features: Number of features in the dataset.

    Returns:
        The updated CNF object with the added data point clauses.
    """
    # Clause (3) and (4): Data point direction based on feature values
    for j in range(num_features):
        Oj = compute_ordering(dataset, j)
        for (i, ip) in Oj:
            if dataset[i][j] < dataset[ip][j]:  # Different feature values (Clause 3)
                for t in branch_nodes:
                    append_direction_clauses(cnf, literals, t, j, i, ip, append_both=False)
            if dataset[i][j] == dataset[ip][j]:  # Equal feature values (Clause 4)
                for t in branch_nodes:
                    append_direction_clauses(cnf, literals, t, j, i, ip, append_both=True)

    cnf = add_path_validity_and_deviation_clauses(cnf, literals, dataset, leaf_nodes)

    return cnf


def construct_feature_selection_clauses(wcnf: Union[WCNF, CNF],  # Use WCNF or CNF as appropriate
                                        literals: Dict[str, int],
                                        dataset: np.ndarray,
                                        branch_nodes: List[int],
                                        leaf_nodes: List[int],
                                        num_features: int) -> Union[WCNF, CNF]:
    """
    Constructs the feature selection clauses for the SAT solver based on the decision tree encoding.

    Args:
        wcnf: The WCNF object to which the clauses will be added.
        literals: A dictionary mapping literals to variable indices.
        dataset: The dataset, a list of tuples representing data points.
        branch_nodes: Indices of branching nodes.
        leaf_nodes: Indices of leaf nodes.
        num_features: Number of features in the dataset.

    Returns:
        A WCNF object containing all the feature selection clauses.
    """
    wcnf = add_feature_selection_clauses_for_branching_nodes(wcnf, literals, branch_nodes, num_features)
    wcnf = add_data_point_clauses(wcnf, literals, dataset, branch_nodes, leaf_nodes, num_features)

    return wcnf


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


def add_redundant_constraints(cnf: Union[WCNF, CNF],  # Use WCNF or CNF as appropriate
                              literals: Dict[str, int],
                              dataset: np.ndarray,
                              branch_nodes: List[int],
                              num_features: int) -> Union[WCNF, CNF]:
    """
    Adds redundant constraints to prune the search space.

    This function adds clauses to ensure that the data point with the lowest feature value is directed left
    and the data point with the highest feature value is directed right for each feature at each branching node.

    Args:
        cnf: The CNF object to which the clauses will be added.
        literals: A dictionary mapping literals to variable indices.
        dataset: The dataset, a list of tuples representing data points.
        branch_nodes: Indices of branching nodes.
        num_features: Number of features in the dataset.

    Returns:
        The updated CNF object with the added redundant constraints.
    """
    # Clause (9) and (10): Redundant constraints to prune the search space
    # These clauses are optimizations
    for t in branch_nodes:
        # Find the data point with the lowest and highest feature value for each feature
        for j in range(num_features):
            sorted_by_feature = sorted(range(len(dataset)), key=lambda k: float(dataset[k][j]))
            lowest_value_index = sorted_by_feature[0]
            highest_value_index = sorted_by_feature[-1]

            # Clause (9): The data point with the lowest feature value is directed left
            cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{lowest_value_index}_{t}']])

            # Clause (10): The data point with the highest feature value is directed right
            cnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{highest_value_index}_{t}']])

    return cnf


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


def add_feature_selection_clauses_for_branching_nodes(cnf: Union[WCNF, CNF],  # Use WCNF or CNF as appropriate
                                                      literals: Dict[str, int],
                                                      branch_nodes: List[int],
                                                      num_features: int) -> CNF:
    """
    Adds feature selection clauses to the CNF object.

    This function adds clauses to ensure that at least one feature is chosen at each branching node
    and no two features are chosen simultaneously.

    Args:
        cnf: The CNF object to which the clauses will be added.
        literals: A dictionary mapping literals to variable indices.
        branch_nodes: Indices of branching nodes.
        num_features: Number of features in the dataset.

    Returns:
        The updated CNF object with the added feature selection clauses.
    """
    # Clause (14) and (15): Feature selection at branching nodes
    for t in branch_nodes:
        # At least one feature is chosen (Clause 15)
        clause = [literals[f'a_{t}_{j}'] for j in range(num_features)]
        cnf.append(clause)

        # No two features are chosen (Clause 14)
        for j in range(num_features):
            for jp in range(j + 1, num_features):
                clause = [-literals[f'a_{t}_{j}'], -literals[f'a_{t}_{jp}']]
                cnf.append(clause)

    return cnf


def add_path_validity_and_deviation_clauses(cnf: CNF,
                                            literals: Dict[str, int],
                                            dataset: np.ndarray,
                                            leaf_nodes: List[int]) -> CNF:
    """
    Adds path validity and deviation clauses to the CNF object.

    This function adds clauses to ensure path validity from right and left traversal,
    and deviations for data points not ending in leaf nodes.

    Args:
        cnf: The CNF object to which the clauses will be added.
        literals: A dictionary mapping literals to variable indices.
        dataset: The dataset, a list of tuples representing data points.
        leaf_nodes: Indices of leaf nodes.

    Returns:
        The updated CNF object with the added path validity and deviation clauses.
    """
    # Clause (5 and 6): Path validity from right traversal and left traversal
    for t in leaf_nodes:
        left_ancestors = get_ancestors(t, 'left')
        right_ancestors = get_ancestors(t, 'right')
        for i in range(len(dataset)):
            if left_ancestors:
                cnf.append([-literals[f'z_{i}_{t}']] + [literals[f's_{i}_{a}'] for a in left_ancestors])
            if right_ancestors:
                cnf.append([-literals[f'z_{i}_{t}']] + [-literals[f's_{i}_{a}'] for a in right_ancestors])

    # Clause (7): Each data point that does not end up in leaf node t has at least one deviation from the path
    for xi in range(len(dataset)):
        for t in leaf_nodes:
            deviations = []
            left_ancestors = get_ancestors(t, 'left')
            right_ancestors = get_ancestors(t, 'right')
            if left_ancestors:
                deviations.extend([-literals[f's_{xi}_{ancestor}'] for ancestor in left_ancestors])
            if right_ancestors:
                deviations.extend([literals[f's_{xi}_{ancestor}'] for ancestor in right_ancestors])
            if deviations:
                cnf.append([literals[f'z_{xi}_{t}']] + deviations)

    return cnf
