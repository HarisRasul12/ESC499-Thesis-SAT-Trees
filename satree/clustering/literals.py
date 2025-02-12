from typing import Any, Dict, List, Tuple, Union

import numpy as np

from satree.treemodder.builder import create_literals


def create_literals_cluster_tree(branch_nodes: List[int],
                                 leaf_nodes: List[int],
                                 feature_indices: np.ndarray,
                                 k_clusters: int,
                                 dataset_size: int,
                                 distance_classes: List[Any],
                                 bicriteria: bool = False) -> Dict[str, int]:
    """
    Generates SAT literals for the clustering encoding by extending the tree-based literal creation with additional
    variables dedicated to cluster assignments and distance-based constraints. In particular, it creates:
      – 'x' literals that encode the ordering of cluster assignments for each data point,
      – 'bw_m' literals to indicate that points within a given distance class should not be clustered together, and
      – optionally, 'bw_p' literals for encouraging co-clustering when bicriteria objectives are considered.
    These additional variables enable the SAT formulation to capture both the hard structural constraints and the
    soft distance-driven preferences central to the clustering mathematics.

    Args:
        branch_nodes: List of indices for branching nodes.
        leaf_nodes: List of indices for leaf nodes.
        feature_indices: An array of feature identifiers.
        k_clusters: The total number of clusters.
        dataset_size: The number of data points in the dataset.
        distance_classes: A collection representing distance classes.
        bicriteria: Flag indicating whether to initialize additional literals for bicriteria objectives.

    Returns:
        A dictionary mapping literal names to their corresponding variable indices.
    """
    C = list(range(k_clusters))  # List of cluster IDs
    literals, current_index = create_literals(branch_nodes, leaf_nodes, feature_indices, C, dataset_size,
                                              fixed_tree=False)

    # Create 'x' The cluster assigned to point 𝑖 is or comes after 𝑐
    for i in range(dataset_size):
        for c in C:
            literals[f'x_{i}_{c}'] = current_index
            current_index += 1

    # Create 'bw_m' literals (points in class w should NOT be clustered together)
    for w, pairs in enumerate(distance_classes):  # data_classes is a list of numpy arrays
        literals[f'bw_m_{w}'] = current_index
        current_index += 1

    if bicriteria:
        # Create 'bw_p' literals (points in class w should be clustered together)
        for w, pairs in enumerate(distance_classes):
            literals[f'bw_p_{w}'] = current_index
            current_index += 1

    return literals


def create_literal_matrices_modular(literals: Dict[str, int],
                                    solution: List[int],
                                    dataset_size: int,
                                    k_clusters: int,
                                    branch_nodes: List[int],
                                    leaf_nodes: List[int],
                                    num_features: int,
                                    distance_classes: List[Any],
                                    bicriteria: bool = False) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Decodes the SAT solution into structured matrices (and vectors) representing truth assignments for the various
    literal groups used in the clustering SAT encoding. The resulting matrices capture:
      – the assignment of features (and thus the tree structure) via 'a', 's', 'z', and 'g' matrices,
      – the cluster assignment ordering through the 'x' matrix, and
      – the activation of distance-based constraints via the 'bw_m' (and optionally 'bw_p') vectors.
    This transformation is essential for bridging the SAT model to an interpretable clustering outcome.

    Args:
        literals: A dictionary mapping literal names to variable indices.
        solution: A collection of literal indices that are True in the SAT model.
        dataset_size: The number of data points in the dataset.
        k_clusters: The total number of clusters.
        branch_nodes: List of indices for branching nodes.
        leaf_nodes: List of indices for leaf nodes.
        num_features: The total number of features.
        distance_classes: A collection representing distance classes.
        bicriteria: If True, an additional matrix for bicriteria objectives is created.

    Returns:
        A tuple containing matrices for:
          - 'a' literals,
          - 's' literals,
          - 'z' literals,
          - 'g' literals,
          - 'x' literals,
          - 'bw_m' vector,
        and if bicriteria is True, also the 'bw_p' vector.
    """

    # Initialize matrices with zeros
    a_matrix = np.zeros((len(branch_nodes), num_features), dtype=int)
    s_matrix = np.zeros((dataset_size, len(branch_nodes)), dtype=int)
    z_matrix = np.zeros((dataset_size, len(leaf_nodes)), dtype=int)
    g_matrix = np.zeros((len(leaf_nodes), k_clusters), dtype=int)
    x_i_c_matrix = np.zeros((dataset_size, k_clusters), dtype=int)
    bw_m_vector = np.zeros(len(distance_classes), dtype=int)
    bw_p_vector = np.zeros(len(distance_classes), dtype=int)  # Always initialize bw_p_vector

    # Helper: update the matrix element based on the literal's index and solution
    def update_matrix(matrix, m, n, literal_index):
        if literal_index in solution:
            matrix[m, n] = 1
        elif -literal_index in solution:
            matrix[m, n] = 0

    # Process literals for matrices: a, s, z, g, and x_i_c.
    for literal, index in literals.items():
        parts = literal.split('_')
        if literal.startswith('a_'):
            # Format: a_{t}_{j}
            t = branch_nodes.index(int(parts[1]))
            j = int(parts[2])
            update_matrix(a_matrix, t, j, index)
        elif literal.startswith('s_'):
            # Format: s_{i}_{t}
            i = int(parts[1])
            t = branch_nodes.index(int(parts[2]))
            update_matrix(s_matrix, i, t, index)
        elif literal.startswith('z_'):
            # Format: z_{i}_{t}
            i = int(parts[1])
            t = leaf_nodes.index(int(parts[2]))
            update_matrix(z_matrix, i, t, index)
        elif literal.startswith('g_'):
            # Format: g_{t}_{c}
            t = leaf_nodes.index(int(parts[1]))
            c = int(parts[2])
            update_matrix(g_matrix, t, c, index)
        elif literal.startswith('x_'):
            # Format: x_{i}_{c}
            i = int(parts[1])
            c = int(parts[2])
            update_matrix(x_i_c_matrix, i, c, index)

    # Process bw_m_ literals
    for literal, index in literals.items():
        if literal.startswith('bw_m_'):
            # Format: bw_m_{w}
            w = int(literal.split('_')[2])
            bw_m_vector[w] = 1 if index in solution else 0

    # If bicriteria is enabled, process bw_p_ literals as well.
    if bicriteria:
        for literal, index in literals.items():
            if literal.startswith('bw_p_'):
                # Format: bw_p_{w}
                w = int(literal.split('_')[2])
                bw_p_vector[w] = 1 if index in solution else 0

    if bicriteria:
        return a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector, bw_p_vector
    else:
        return a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector
