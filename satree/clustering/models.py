"""
=========== Module Description ===========

This module implements the SAT-based encoding for clustering problems using fixed-depth tree structures.
It constructs weighted CNF formulations that integrate the base tree encoding with clustering-specific constraints,
including:
  • Must-link and cannot-link constraints that capture pairwise clustering relationships,
  • Distance class constraints derived from the computed distance classes to enforce soft penalties based on
    intra-cluster distances.
The encoding transforms the clustering objective into a partial MaxSAT problem where soft clauses are used to
balance cluster cohesion (by grouping close points) and separation (by penalizing clusters that merge distant points).
Multiple encoding variants are provided, including an enhanced version that incorporates “smart pair” constraints
to further refine clustering performance.
"""

from typing import Dict, List

import numpy as np
from pysat.formula import WCNF
from scipy.spatial.distance import euclidean

from satree.clustering.sat_clauses import construct_clustering_clauses, add_clustering_encodings, \
    add_distance_class_clauses


def build_clauses_cluster_tree_md_ms(literals: Dict[str, int],
                                     dataset: np.ndarray,
                                     branch_nodes: List[int],
                                     leaf_nodes: List[int],
                                     num_features: int,
                                     k_clusters: int,
                                     cl_pairs: np.ndarray,
                                     ml_pairs: np.ndarray,
                                     distance_classes: List[np.ndarray]) -> WCNF:
    """
    Constructs the weighted CNF encoding for a clustering problem based on a fixed-depth tree structure and a
    minimum split criterion. It integrates the base tree encoding with additional clustering constraints that incorporate:
      – must-link and cannot-link pairs (reflecting pairwise clustering relationships), and
      – distance class constraints to enforce soft penalties based on intra-class distances.
    This formulation transforms the clustering objective into a partial MaxSAT problem where the soft clauses drive the
    optimization of cluster cohesion and separation.

    Args:
        literals: A dictionary mapping literal names to variable indices.
        dataset: The dataset containing data points.
        branch_nodes: Indices of branching nodes.
        leaf_nodes: Indices of leaf nodes.
        num_features: Number of features in the dataset.
        k_clusters: The number of clusters.
        cl_pairs: Array of cannot-link pairs.
        ml_pairs: Array of must-link pairs.
        distance_classes: List of arrays for each distance class.

    Returns:
        A weighted CNF object encoding the clustering clauses.
    """
    ##################################################  BASE TREE ENCODINGS ################################################

    # Now the problem has become Partial MaxSAT - we will assign weights to the soft clauses Eq. (13). Eq(1-10,12) HARD clauses
    wcnf = construct_clustering_clauses(literals, dataset, branch_nodes, leaf_nodes, num_features)
    wcnf = add_clustering_encodings(wcnf, literals, dataset, leaf_nodes, k_clusters, cl_pairs, ml_pairs,
                                    distance_classes)
    wcnf = add_distance_class_clauses(wcnf, literals, k_clusters, distance_classes)

    return wcnf


def build_clauses_cluster_tree_md_ms_smart_pair(literals: Dict[str, int],
                                                dataset: np.ndarray,
                                                branch_nodes: List[int],
                                                leaf_nodes: List[int],
                                                num_features: int,
                                                k_clusters: int,
                                                cl_pairs: np.ndarray,
                                                ml_pairs: np.ndarray,
                                                distance_classes: List[np.ndarray]) -> WCNF:
    """
    Constructs an enhanced SAT encoding for clustering that incorporates “smart pair” constraints. In this variant,
    must-link and cannot-link pairs are pre-sorted based on Euclidean distance so that the SAT clauses can be added
    conditionally—ensuring that pairs with closer proximity are preferentially forced into the same cluster and
    those farther apart are separated. Additionally, distance class clauses are integrated to conditionally require
    co-clustering or separation, thus refining the optimization landscape as defined by the clustering mathematics.

    Args:
        literals: A dictionary mapping literal names to variable indices.
        dataset: The dataset containing data points.
        branch_nodes: List of indices for branching nodes.
        leaf_nodes: List of indices for leaf nodes.
        num_features: The number of features in the dataset.
        k_clusters: The number of clusters.
        cl_pairs: Array of cannot-link pairs.
        ml_pairs: Array of must-link pairs.
        distance_classes: List of arrays representing distance classes.

    Returns:
        A weighted CNF object containing clustering clauses that integrate smart pair constraints.
    """
    ##################################################  BASE TREE ENCODINGS ################################################

    wcnf = construct_clustering_clauses(literals, dataset, branch_nodes, leaf_nodes, num_features)

    ################################################## CLUSTERING PROBLEM ENCODINGS ################################################

    # Smart Pairs initialization
    E_plus = set()
    E_minus = set()

    # Process must-link pairs
    ML_pairs_sorted = sorted(ml_pairs, key=lambda pair: euclidean(dataset[pair[0]], dataset[pair[1]]))
    for (i, i_prime) in ML_pairs_sorted:
        if (i, i_prime) not in E_plus:
            E_plus.add((i, i_prime))
            # Add must-link clauses
            for c in range(k_clusters - 1):
                wcnf.append([-literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']])
                wcnf.append([literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])

    # Process cannot-link pairs
    CL_pairs_sorted = sorted(cl_pairs, key=lambda pair: -euclidean(dataset[pair[0]], dataset[pair[1]]))
    for (i, i_prime) in CL_pairs_sorted:
        if (i, i_prime) not in E_minus:
            E_minus.add((i, i_prime))
            # Add cannot-link clauses
            for c in range(k_clusters - 1):
                wcnf.append([-literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])

    # Process distance classes
    for w, pairs in enumerate(distance_classes):
        for (i, i_prime) in pairs:
            if (i, i_prime) in E_plus:
                continue
            if (i, i_prime) in E_minus:
                wcnf.append([literals[f'bw_m_{w}']])
            else:
                # Add conditional clauses for co-clustering
                wcnf.append([literals[f'bw_m_{w}'], literals[f'x_{i}_0'], literals[f'x_{i_prime}_0']])
                wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{k_clusters - 2}'],
                             -literals[f'x_{i_prime}_{k_clusters - 2}']])
                c = 0  # Ensure c is defined even if the loop doesn't execute
                for c in range(k_clusters - 2):
                    wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}'],
                                 literals[f'x_{i}_{c + 1}'], literals[f'x_{i_prime}_{c + 1}']])
                wcnf.append([-literals[f'bw_p_{w}'], -literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']])
                wcnf.append([-literals[f'bw_p_{w}'], literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])

    wcnf = add_clustering_encodings(wcnf, literals, dataset, leaf_nodes, k_clusters, cl_pairs, ml_pairs,
                                    distance_classes)
    wcnf = add_distance_class_clauses(wcnf, literals, k_clusters, distance_classes)

    return wcnf


def build_clauses_cluster_tree_md(literals: Dict[str, int],
                                  dataset: np.ndarray,
                                  branch_nodes: List[int],
                                  leaf_nodes: List[int],
                                  num_features: int,
                                  k_clusters: int,
                                  cl_pairs: np.ndarray,
                                  ml_pairs: np.ndarray,
                                  distance_classes: List[np.ndarray]) -> WCNF:
    """
    Constructs a SAT encoding for clustering using a fixed-depth tree without the enhanced split criteria. This
    function merges the base tree encoding with clustering-specific constraints, including those that enforce
    non-empty clusters and proper separation via distance class constraints. The generated CNF reflects a set of
    both hard and soft clauses that directly correspond to the clustering optimization problem—balancing cohesion
    (by grouping close points) and separation (by penalizing clusters that merge distant points).

    Args:
        literals: A dictionary mapping literal names to variable indices.
        dataset: The dataset containing data points.
        branch_nodes: Indices of branching nodes.
        leaf_nodes: Indices of leaf nodes.
        num_features: Number of features in the dataset.
        k_clusters: The number of clusters.
        cl_pairs: Cannot-link pairs.
        ml_pairs: Must-link pairs.
        distance_classes: List of arrays where each array contains point index pairs for a distance class.

    Returns:
        A weighted CNF object encoding the clustering constraints, including both hard and soft clauses.
    """
    ##################################################  BASE TREE ENCODINGS ################################################

    # Now the problem has become Partial MaxSAT - we will assign weights to the soft clauses Eq. (13). Eq(1-10,12) HARD clauses
    wcnf = construct_clustering_clauses(literals, dataset, branch_nodes, leaf_nodes, num_features)
    wcnf = add_clustering_encodings(wcnf, literals, dataset, leaf_nodes, k_clusters, cl_pairs, ml_pairs,
                                    distance_classes)

    # Clause 32: Ensures that if bw^-_w is true, then the points in distance class w
    # cannot be clustered with the points in distance class w-1 if bw^-_(w-1) is false.
    for w in range(1, len(distance_classes)):  # Starting from 1 since we're checking w against w-1
        # print('clause 32: ', [-literals[f'bw_m_{w}'], literals[f'bw_m_{w-1}']])
        wcnf.append([-literals[f'bw_m_{w}'], literals[f'bw_m_{w - 1}']])

    # Clause 37: For each distance class w, we add a soft clause for the corresponding b^-_w literal
    # to encourage points within that class to be clustered separately
    # Max diameter solve problem
    for w in range(len(distance_classes)):
        # print("clause 37: ", [-literals[f'bw_m_{w}']])
        wcnf.append([-literals[f'bw_m_{w}']], weight=1)

    return wcnf
