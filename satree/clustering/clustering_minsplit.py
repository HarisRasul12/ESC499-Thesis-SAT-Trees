"""
=========== Module Description ===========

This module contains the functions to solve the clustering minimum split problem using a SAT solver.
"""

from typing import Dict, List, Tuple

import numpy as np
from pysat.formula import WCNF

from satree.treemodder.builder import build_complete_tree

from satree.clustering.core import create_literals_cluster_tree, create_literal_matrices_modular
from satree.clustering.clustering_advanced import solve_wcnf_clustering, create_distance_classes, \
    assign_clusters_and_diameters
from satree.clustering.clustering_clauses import construct_clustering_clauses, add_clustering_encodings, \
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
    Construct clustering clauses for the SAT solver using a fixed-depth tree encoding with minimum split criteria.

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


def process_clustering_solution(wcnf: WCNF,
                                literals: Dict[str, int],
                                dataset: np.ndarray,
                                features: np.ndarray,
                                k_clusters: int,
                                branch_nodes: List[int],
                                leaf_nodes: List[int],
                                distance_classes: List[np.ndarray]) -> Tuple[
    Dict[int, List[int]], Dict[int, float], List[int]]:
    """
    Solve the clustering SAT problem and process the solution to obtain cluster assignments and diameters.

    Args:
        wcnf: The weighted CNF object containing clustering clauses.
        literals: A dictionary mapping literal names to variable indices.
        dataset: The dataset containing data points.
        features: An array of feature identifiers.
        k_clusters: The total number of clusters.
        branch_nodes: Indices of branching nodes.
        leaf_nodes: Indices of leaf nodes.
        distance_classes: List of arrays for each distance class.

    Returns:
        A tuple containing:
          - A dictionary mapping cluster IDs to lists of data point indices.
          - A dictionary mapping cluster IDs to the maximum diameter of each cluster.
          - The SAT solver's solution as a list of literal indices.
    """
    solution = solve_wcnf_clustering(wcnf)

    a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector, bw_p_vector = create_literal_matrices_modular(
        literals=literals,
        solution=solution,
        dataset_size=len(dataset),
        k_clusters=k_clusters,
        branch_nodes=branch_nodes,
        leaf_nodes=leaf_nodes,
        num_features=len(features),
        distance_classes=distance_classes,
        bicriteria=True
    )

    cluster_assignments, cluster_diameters = assign_clusters_and_diameters(
        x_i_c_matrix, dataset, k_clusters
    )

    return cluster_assignments, cluster_diameters, solution


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
        depth: The depth of the complete binary tree used for clustering.
        epsilon: Maximum distance difference to consider distances similar.
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
    wcnf = build_clauses_cluster_tree_md_ms(literals, dataset, TB, TL, num_features, k_clusters,
                                            cl_pairs, ml_pairs, distance_classes)

    return process_clustering_solution(wcnf, literals, dataset, features, k_clusters, TB, TL, distance_classes)[:2]
