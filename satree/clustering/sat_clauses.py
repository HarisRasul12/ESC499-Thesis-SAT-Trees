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
    Builds the initial set of SAT clauses for the clustering problem by adapting the tree-based feature selection
    and redundant constraints from decision tree encoding to the clustering context. This foundational encoding
    guarantees a valid tree structure while providing a framework upon which clustering-specific clauses can be added.

    Args:
        literals: A dictionary mapping literal names to variable indices.
        dataset: The dataset containing data points.
        branch_nodes: Indices of branching nodes.
        leaf_nodes: Indices of leaf nodes.
        num_features: Number of features in the dataset.

    Returns:
        A weighted CNF object containing clustering clauses (both hard and soft constraints).
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
    Augments the base SAT encoding with additional clauses that enforce clustering properties. These include unary
    ordering of cluster labels at leaves, constraints linking data point routing to cluster assignments, and clauses
    that integrate must-link and cannot-link conditions. This expanded encoding is designed to ensure that the SAT
    solution aligns with the mathematical objectives of clustering—maximizing intra-cluster similarity while enforcing
    inter-cluster separation.

    Args:
        wcnf: The weighted CNF object to update.
        literals: A dictionary mapping literal names to variable indices.
        dataset: The dataset containing data points.
        leaf_nodes: Indices of leaf nodes.
        k_clusters: Total number of clusters.
        cl_pairs: Array of cannot-link pairs.
        ml_pairs: Array of must-link pairs.
        distance_classes: List of arrays representing distance classes.

    Returns:
        The updated weighted CNF object with clustering encoding clauses added.
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
    Integrates distance class constraints into the SAT formulation by adding clauses that condition the clustering
    decisions on the pairwise distance groupings. These clauses ensure that, for each distance class, either:
      – data points are forced to be clustered together if the corresponding “bw_p” indicator is true, or
      – separated if the “bw_m” indicator is active.
    Soft clauses are also added to encourage desirable clustering outcomes (e.g., minimizing cluster diameter),
    reflecting the balance between cohesion and separation central to the mathematical model.

    Args:
        wcnf: The weighted CNF object to update.
        literals: A dictionary mapping literal names to variable indices.
        k_clusters: The total number of clusters.
        distance_classes: List of arrays, each representing a distance class.

    Returns:
        The updated weighted CNF object with distance class clauses.
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
