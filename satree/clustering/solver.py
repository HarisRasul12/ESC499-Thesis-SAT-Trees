"""
=========== Module Description ===========

This module provides the end-to-end SAT solving and post-processing pipeline for the clustering problem.
It employs a Partial MaxSAT solver (RC2) to find an assignment that satisfies the weighted CNF formulation
constructed from the clustering constraints. After solving, the module decodes the SAT solution into structured
literal matrices that represent cluster assignments and distance class indicators. It then interprets these matrices
to:
  • Assign data points to clusters based on unique patterns in the cluster assignment matrix,
  • Compute key clustering metrics such as the maximum diameter within each cluster.
This process bridges the abstract SAT encoding with practical clustering outcomes, ensuring that the solution
aligns with the optimization objectives of achieving cohesive, well-separated clusters as outlined in the mathematical model.
"""

from typing import List, Tuple, Dict

import numpy as np
from scipy.spatial.distance import euclidean
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from satree.clustering.literals import create_literal_matrices_modular


def solve_wcnf_clustering(wcnf: WCNF) -> List[int]:
    """
    Solves the weighted CNF clustering formulation using a Partial MaxSAT solver. The solution—a list of literal
    assignments—represents an optimized assignment that satisfies the hard constraints (e.g., tree validity, pairwise
    constraints) while optimizing the soft clustering objectives (e.g., intra-cluster compactness).

    Args:
        wcnf: The weighted CNF object containing the clustering clauses.

    Returns:
        A list of literal indices representing the SAT model, or an empty list if no solution was found.
    """
    solver = RC2(wcnf)
    solution = solver.compute()
    return solution if solution is not None else []


def assign_clusters_and_diameters(x_i_c_matrix: np.ndarray,
                                  dataset: np.ndarray,
                                  k_clusters: int) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
    """
    Interprets the cluster assignment matrix obtained from the SAT solution to assign each data point to a cluster
    and computes the maximum diameter (largest pairwise distance) within each cluster. This post-processing step translates
    the SAT model into actionable clustering results, quantifying both the grouping and quality (via diameter) of each cluster.

    Args:
        x_i_c_matrix: A matrix representing cluster assignments for data points.
        dataset: The original dataset with data points.
        k_clusters: The total number of clusters.

    Returns:
        A tuple containing:
          - A dictionary mapping cluster IDs to lists of data point indices.
          - A dictionary mapping cluster IDs to the maximum diameter (largest pairwise distance) within that cluster.
    """
    # Assign clusters based on unique patterns in the x_i_c_matrix
    unique_patterns = np.unique(x_i_c_matrix, axis=0)
    pattern_to_cluster = {tuple(pattern): cluster_id for cluster_id, pattern in enumerate(unique_patterns)}

    cluster_assignments = {cluster_id: [] for cluster_id in range(k_clusters)}
    for data_point_index, pattern in enumerate(x_i_c_matrix):
        cluster_id = pattern_to_cluster[tuple(pattern)]
        cluster_assignments[cluster_id].append(data_point_index)

    # Calculate the maximum diameter for each cluster
    cluster_diameters = {}
    for cluster_id, data_points in cluster_assignments.items():
        max_diameter = 0
        # Calculate all pairwise distances within the cluster
        for i in range(len(data_points)):
            for j in range(i + 1, len(data_points)):
                dist = euclidean(dataset[data_points[i]], dataset[data_points[j]])
                max_diameter = max(max_diameter, dist)
        cluster_diameters[cluster_id] = max_diameter

    return cluster_assignments, cluster_diameters


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
    Integrates the entire SAT-based clustering pipeline: it solves the weighted CNF formulation, decodes the solution
    into modular literal matrices (including cluster assignment and distance class indicators), and then interprets these
    matrices to derive final cluster assignments and compute cluster diameters. This function encapsulates the end-to-end
    process of translating the SAT model into a clustering outcome as defined by the mathematical formulation.

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
