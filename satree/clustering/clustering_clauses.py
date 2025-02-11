from typing import Dict, List

import numpy as np
from pysat.formula import WCNF

from satree.common_sat_clauses import construct_feature_selection_clauses, add_redundant_constraints


def construct_clustering_clauses(literals: Dict[str, int],
                                 dataset: np.ndarray,
                                 branch_nodes: List[int],
                                 leaf_nodes: List[int],
                                 num_features: int) -> WCNF:
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding for clustering.

    Args:
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        branch_nodes (list): Indices of branching nodes.
        leaf_nodes (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.

    Returns:
        WCNF: A WCNF object containing all the clauses, with hard clauses for the tree structure and soft clauses for maximizing correctly classified points.
    """
    wcnf = WCNF()
    wcnf = construct_feature_selection_clauses(wcnf, literals, dataset, branch_nodes, leaf_nodes, num_features)
    wcnf = add_redundant_constraints(wcnf, literals, dataset, branch_nodes, num_features)

    return wcnf


def add_clustering_encodings(wcnf: WCNF,
                             literals: Dict[str, int],
                             dataset: np.ndarray,
                             leaf_nodes: List[int],
                             k_clusters: int,
                             cl_pairs: np.ndarray,
                             ml_pairs: np.ndarray,
                             distance_classes: List[np.ndarray]) -> WCNF:
    """
    Adds clustering clauses to the WCNF object.

    This function adds various clauses to ensure proper clustering, including unary encoding of cluster labels,
    assignment of data points to clusters, and constraints for must-link and cannot-link pairs.

    Args:
        wcnf (WCNF): The WCNF object to which the clauses will be added.
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        leaf_nodes (list): Indices of leaf nodes.
        k_clusters (int): Number of clusters.
        cl_pairs (list): Cannot-link pairs.
        ml_pairs (list): Must-link pairs.
        distance_classes (list): List of pairs in each distance class.

    Returns:
        WCNF: The updated WCNF object with the added clustering clauses.
    """
    # Clause 16: Unary encoding of cluster labels in each leaf
    for t in leaf_nodes:
        for c in range(k_clusters - 2):
            clause = [literals[f'g_{t}_{c}'], -literals[f'g_{t}_{c + 1}']]
            wcnf.append(clause)

    # Clause 17: Data points ending at leaf node t are assigned to cluster c if g_t,c is true
    for t in leaf_nodes:
        for i in range(len(dataset)):
            for c in range(k_clusters - 1):
                clause = [-literals[f'z_{i}_{t}'], -literals[f'g_{t}_{c}'], literals[f'x_{i}_{c}']]
                wcnf.append(clause)

    # Clause 18: Data points ending at leaf node t are NOT assigned to cluster c if g_t,c is false
    for t in leaf_nodes:
        for i in range(len(dataset)):
            for c in range(k_clusters - 1):
                clause = [-literals[f'z_{i}_{t}'], literals[f'g_{t}_{c}'], -literals[f'x_{i}_{c}']]
                wcnf.append(clause)

    # Clause 19: Ensure no cluster is empty by ensuring there's at least one data point in each cluster
    for c in range(k_clusters - 1):
        wcnf.append([-literals[f'x_{c}_{c}']])

    # Clause 20: If xi is not in cluster c, then there must be some xi' in cluster c-1, for all c < i
    for i in range(1, len(dataset)):
        for c in range(1, k_clusters - 1):
            clause = [-literals[f'x_{i}_{c}']]
            for i_prime in range(i):
                clause.append(literals[f'x_{i_prime}_{c - 1}'])
            wcnf.append(clause)

    # Clause 21: Ensure all clusters are non-empty by requiring at least one point is assigned to each cluster
    clauseTW = [literals[f'x_{i}_{k_clusters - 2}'] for i in range(len(dataset))]
    wcnf.append(clauseTW)

    # Clause 22: Ensure that pairs in CL are not clustered in the first cluster (0-indexed)
    for i, i_prime in cl_pairs:
        wcnf.append([literals[f'x_{i}_0'], literals[f'x_{i_prime}_0']])

    # Clause 23: Ensure that pairs in CL are not clustered in the last cluster (k-2 in 0-indexed system)
    for i, i_prime in cl_pairs:
        wcnf.append([-literals[f'x_{i}_{k_clusters - 2}'], -literals[f'x_{i_prime}_{k_clusters - 2}']])

    # Clause 24: Unconditional separating clauses for cannot-link pairs, applied to clusters from 0 to k-3
    for (i, i_prime) in cl_pairs:
        for c in range(k_clusters - 2):
            wcnf.append([
                -literals[f'x_{i}_{c}'],
                -literals[f'x_{i_prime}_{c}'],
                literals[f'x_{i}_{c + 1}'],
                literals[f'x_{i_prime}_{c + 1}']
            ])

    # Clause 25 and 26: Ensure that pairs in ML are clustered together for each cluster
    for i, i_prime in ml_pairs:
        for c in range(k_clusters - 1):
            wcnf.append([-literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']])  # clause 25
            wcnf.append([literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])  # clause 26

    # Clause 27: Conditional separating clauses using distance classes and bw_m literals
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            wcnf.append([literals[f'bw_m_{w}'], literals[f'x_{i}_0'], literals[f'x_{i_prime}_0']])

    # Clause 28: Ensure that if bw^-_w is true, then the pair (i, i') from Dw cannot be in the second to last cluster k-2
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{k_clusters - 2}'],
                         -literals[f'x_{i_prime}_{k_clusters - 2}']])

    # Clause 29: Conditional co-separation for non-adjacent clusters
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            for c in range(k_clusters - 2):
                wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{c}'],
                             -literals[f'x_{i_prime}_{c}'], literals[f'x_{i}_{c + 1}'],
                             literals[f'x_{i_prime}_{c + 1}']])

    return wcnf


def add_distance_class_clauses(wcnf: WCNF,
                               literals: Dict[str, int],
                               k_clusters: int,
                               distance_classes: List[np.ndarray]) -> WCNF:
    """
    Adds distance class clauses to the WCNF object.

    This function adds various clauses to ensure proper clustering based on distance classes,
    including constraints for must-link and cannot-link pairs within distance classes.

    Args:
        wcnf (WCNF): The WCNF object to which the clauses will be added.
        literals (dict): A dictionary mapping literals to variable indices.
        k_clusters (int): Number of clusters.
        distance_classes (list): List of pairs in each distance class.

    Returns:
        WCNF: The updated WCNF object with the added distance class clauses.
    """
    # Clause 30: If b^+_w is true, then pairs (i, i') in distance class w must be in the same cluster
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            for c in range(k_clusters - 1):
                wcnf.append([-literals[f'bw_p_{w}'], -literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']])

    # Clause 31: If b^+_w is true, then points (i, i') in distance class w must be in the same cluster c
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            for c in range(k_clusters - 1):
                wcnf.append([-literals[f'bw_p_{w}'], literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])

    # Clause 32: Ensures that if bw^-_w is true, then the points in distance class w
    # cannot be clustered with the points in distance class w-1 if bw^-_(w-1) is false.
    for w in range(1, len(distance_classes)):
        wcnf.append([-literals[f'bw_m_{w}'], literals[f'bw_m_{w - 1}']])

    # Clause 33: If bw^+_w is true, then distance class w must be clustered together with distance class w-1
    for w in range(1, len(distance_classes)):
        wcnf.append([-literals[f'bw_p_{w}'], literals[f'bw_p_{w - 1}']])

    # Clause 34: If bw^+_w is true, then distance class w cannot be clustered separately within itself
    for w in range(len(distance_classes)):
        wcnf.append([-literals[f'bw_p_{w}'], literals[f'bw_m_{w}']])

    # Clause 37: For each distance class w, we add a soft clause for the corresponding b^-_w literal
    # to encourage points within that class to be clustered separately
    for w in range(len(distance_classes)):
        wcnf.append([-literals[f'bw_m_{w}']], weight=1)

    # Clause 38: For each distance class w, we add a soft clause for the corresponding b^+_w literal
    # to encourage points within that class to be clustered together
    for w in range(len(distance_classes)):
        wcnf.append([literals[f'bw_p_{w}']], weight=1)

    return wcnf
