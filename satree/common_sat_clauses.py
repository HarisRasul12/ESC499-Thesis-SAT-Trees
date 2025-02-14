"""
=========== Module Description ===========

common_sat_clauses.py

This module implements the core functions for constructing SAT clauses that encode the decision tree’s structure and
data routing constraints. It provides a set of tools to translate a decision tree learning problem into a CNF or
weighted CNF (WCNF) formulation that a SAT solver can process.

Key functionalities include:
1. Feature Selection Clauses:
   - Builds clauses ensuring that each branching node in the decision tree selects exactly one feature.
   - Prevents multiple feature selections at the same node.

2. Data Point Clauses:
   - Generates clauses that direct data points through the tree based on their feature values.
   - Enforces consistent routing by comparing the order of feature values for each data point.

3. Redundant Constraints:
   - Adds extra constraints to optimize the search space.
   - Forces data points with the lowest or highest feature values to follow predetermined paths, reducing ambiguity.

4. Path Validity and Deviations:
   - Ensures that the decision path from the root to each leaf is valid.
   - Introduces deviation clauses to handle cases where data points do not perfectly follow a designated path.

Additional helper functions include:
- Computing the ordering of data points for each feature.
- Appending directional clauses for specific branch nodes and features.

This module is essential for converting the decision tree learning problem into a SAT formulation, enabling objectives
like achieving 100% training accuracy with a minimum height tree or maximizing accuracy within a fixed depth.
"""

from typing import Union, Dict, List, Tuple

import numpy as np
from pysat.formula import WCNF, CNF

from satree.treemodder.tree_utils import get_ancestors


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


def compute_ordering(dataset: np.ndarray,
                     feature_index: int) -> List[Tuple[int, int]]:
    """
    Computes the ordering of data point indices for a specified feature.

    The function sorts the data points based on the value of the specified feature and returns a list of tuples,
    each containing a pair of consecutive data point indices from the sorted order.

    Args:
        dataset: A list of data points (each data point can be a tuple or list).
        feature_index: The index of the feature used for sorting.

    Returns:
        A list of tuples, where each tuple contains a pair (i, j) representing consecutive data point indices
                    in the sorted order.
    """
    sorted_indices = sorted(range(len(dataset)), key=lambda i: float(dataset[i][feature_index]))
    return [(sorted_indices[i], sorted_indices[i + 1]) for i in range(len(sorted_indices) - 1)]


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
