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
    Constructs the clauses for the SAT solver based on the decision tree encoding. Now includes MAX SOLVER PROBLEM FOR FIXED HEIGHT 

    Args:
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        branch_nodes (list): Indices of branching nodes.
        leaf_nodes (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.
        k_clusters: number of clusters, will need to turn this into a list for operations on each clause
        cl_pairs (list): Cannot-link pairs.
        ml_pairs (list): Must-link pairs.
        distance_classes (list): list pairs in ecah distace classes  

    Returns:
        wcnf: A wcnf object containing all the clauses, with hard clauses for the tree structure/clustering and soft clauses for maximization bicriteria
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
    Processes the clustering solution by solving the WCNF problem and creating literal matrices.

    Args:
        wcnf (WCNF): The WCNF object containing the clauses.
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        features (list): List of feature names or indices.
        k_clusters (int): Number of clusters.
        branch_nodes (list): Indices of branching nodes.
        leaf_nodes (list): Indices of leaf nodes.
        distance_classes (list): List of pairs in each distance class.

    Returns:
        tuple: A tuple containing cluster assignments and cluster diameters.
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
    Solves a clustering minimum split problem by constructing a complete binary tree of a specified depth,
    creating literals for a SAT solver, building clauses for the SAT problem, and then solving
    the weighted CNF problem to determine the cluster assignments and the maximum diameter
    of each cluster.

    Args:
        dataset (np.ndarray): The dataset containing n-dimensional data points.
        features (np.ndarray): Array of feature names or indices.
        k_clusters (int): The desired number of clusters to form.
        depth (int): The depth of the complete binary tree for clustering.
        epsilon (float, optional): The maximum distance difference to consider two distances as similar, defaults to 0.
        cl_pairs (np.ndarray, optional): An array of data point pairs that cannot be in the same cluster (cannot-link constraints).
        ml_pairs (np.ndarray, optional): An array of data point pairs that must be in the same cluster (must-link constraints).

    Returns:
    - cluster_assignments (dict): A dictionary with keys as cluster IDs and values as lists of data points in each cluster.
    - cluster_diameters (dict): A dictionary with keys as cluster IDs and values as the maximum diameter of each cluster.

    The function performs the following steps:
    - Creates non-overlapping distance classes for all unique pairs of data points in the dataset.
    - Constructs a complete binary tree for the given depth and assigns branching and leaf nodes.
    - Generates literals required for the SAT solver based on the tree structure and dataset.
    - Builds clauses for the SAT solver based on the decision tree encoding.
    - Solves the weighted CNF problem to find a solution for the clustering.
    - Assigns data points to clusters based on the solution and calculates the maximum diameter for each cluster.
    """
    dataset_size = len(dataset)
    num_features = len(features)
    dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
    tree_structure, TB, TL = build_complete_tree(depth)
    literals = create_literals_cluster_tree(TB, TL, features, k_clusters, dataset_size, distance_classes, True)
    wcnf = build_clauses_cluster_tree_md_ms(literals, dataset, TB, TL, num_features, k_clusters,
                                            cl_pairs, ml_pairs, distance_classes)

    return process_clustering_solution(wcnf, literals, dataset, features, k_clusters, TB, TL, distance_classes)[:2]
