"""
=========== Module Description ===========

This module contains functions to solve the clustering problem using a minimum height tree with a maximum diameter and
minimum split criteria.
"""

from typing import Dict, List

import numpy as np
from scipy.spatial.distance import euclidean
from pysat.formula import WCNF

from satree.treemodder.builder import build_complete_tree

from satree.clustering.core import create_literals_cluster_tree
from satree.clustering.clustering_advanced import create_distance_classes
from satree.clustering.clustering_clauses import add_clustering_encodings, add_distance_class_clauses, \
    construct_clustering_clauses
from satree.clustering.clustering_minsplit import process_clustering_solution


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
    Construct clustering clauses for the SAT solver that integrate smart pair constraints.

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


def min_split_clustering_problem_smart_pair(dataset: np.ndarray,
                                            features: np.ndarray,
                                            k_clusters: int,
                                            depth: int,
                                            epsilon: float = 0,
                                            cl_pairs: np.ndarray = np.array([]),
                                            ml_pairs: np.ndarray = np.array([])) -> tuple:
    """
    Solve the clustering minimum split problem with smart pair constraints using a SAT solver.

    Args:
        dataset: The dataset containing n-dimensional data points.
        features: An array of feature identifiers.
        k_clusters: The desired number of clusters.
        depth: The depth of the complete binary tree for clustering.
        epsilon: The maximum distance difference to consider distances similar.
        cl_pairs: Array of cannot-link pairs.
        ml_pairs: Array of must-link pairs.

    Returns:
        A tuple containing:
          - A dictionary mapping cluster IDs to lists of data point indices.
          - A dictionary mapping cluster IDs to the maximum diameter of each cluster.
    """
    dataset_size = len(dataset)
    num_features = len(features)
    dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
    tree_structure, TB, TL = build_complete_tree(depth)
    literals = create_literals_cluster_tree(TB, TL, features, k_clusters, dataset_size, distance_classes, True)
    wcnf = build_clauses_cluster_tree_md_ms_smart_pair(literals, dataset, TB, TL, num_features, k_clusters,
                                                       cl_pairs, ml_pairs, distance_classes)

    return process_clustering_solution(wcnf, literals, dataset, features, k_clusters, TB, TL, distance_classes)[:2]
