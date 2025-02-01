import numpy as np

from satree.treemodder.builder import create_literals


def create_literals_cluster_tree(TB, TL, F, k_clusters, dataset_size, distance_classes, bicriteria=False):
    """
    Create the literals for the SAT solver based on the tree structure and dataset size.

    This function creates four types of literals:
    - 'a' literals for feature splits at branching nodes,
    - 's' literals for data points directed to left or right,
    - 'z' literals for data points that end up at a leaf node,
    - 'g' literals for assigning class labels to leaf nodes.
    - 'x' The cluster assigned to point 𝑖 is or comes after 𝑐lass label c
    - 'bw_p' The pairs in class 𝑤 should be clustered together
    - 'bw_m' (The negation of) whether the pairs in distance class 𝑤 should be clustered separately

    Parameters:
    - TB (list): Indices of branching nodes in the tree.
    - TL (list): Indices of leaf nodes in the tree.
    - F (list): The array of features
    - k_clusters (int) - based on clusters , so we need to turn this into arrat: eg C= k_clusters, C =2, C-> [0,1], turn into list of cluster ids
    - dataset_size (int): The number of data points in the dataset.
    - data_classes


    Returns:
    - literals (dict): A dictionary where keys are literal names and values are their corresponding indices for the SAT solver.
    """
    C = list(range(k_clusters))  # List of cluster IDs
    literals, current_index = create_literals(TB, TL, F, C, dataset_size, fixed_tree=False)

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


def create_literal_matrices_modular(literals, solution, dataset_size, k_clusters, TB, TL, num_features,
                                    distance_classes, bicriteria=False):
    """
    Constructs literal matrices from the given literal dictionary and solution.

    Parameters
    ----------
    literals : dict
        Dictionary mapping literal names (e.g., 'a_1_2', 's_3_4', etc.) to indices.
    solution : iterable
        A collection of literal indices that are considered 'true' in the solution.
    dataset_size : int
        Number of data points in the dataset.
    k_clusters : int
        Total number of clusters.
    TB : list
        List of indices for branching nodes.
    TL : list
        List of indices for leaf nodes.
    num_features : int
        Total number of features.
    distance_classes : list or similar
        Collection used to determine the number of distance classes.
    bicriteria : bool, optional
        If True, an additional vector bw_p_vector is initialized, processed, and returned.

    Returns
    -------
    If bicriteria is False, returns:
        a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector
    If bicriteria is True, returns:
        a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector, bw_p_vector
    """

    # Initialize matrices with zeros
    a_matrix = np.zeros((len(TB), num_features), dtype=int)
    s_matrix = np.zeros((dataset_size, len(TB)), dtype=int)
    z_matrix = np.zeros((dataset_size, len(TL)), dtype=int)
    g_matrix = np.zeros((len(TL), k_clusters), dtype=int)
    x_i_c_matrix = np.zeros((dataset_size, k_clusters), dtype=int)
    bw_m_vector = np.zeros(len(distance_classes), dtype=int)
    if bicriteria:
        bw_p_vector = np.zeros(len(distance_classes), dtype=int)

    # Helper: update the matrix element based on the literal's index and solution
    def update_matrix(matrix, i, j, literal_index):
        if literal_index in solution:
            matrix[i, j] = 1
        elif -literal_index in solution:
            matrix[i, j] = 0

    # Process literals for matrices: a, s, z, g, and x_i_c.
    for literal, index in literals.items():
        parts = literal.split('_')
        if literal.startswith('a_'):
            # Format: a_{t}_{j}
            t = TB.index(int(parts[1]))
            j = int(parts[2])
            update_matrix(a_matrix, t, j, index)
        elif literal.startswith('s_'):
            # Format: s_{i}_{t}
            i = int(parts[1])
            t = TB.index(int(parts[2]))
            update_matrix(s_matrix, i, t, index)
        elif literal.startswith('z_'):
            # Format: z_{i}_{t}
            i = int(parts[1])
            t = TL.index(int(parts[2]))
            update_matrix(z_matrix, i, t, index)
        elif literal.startswith('g_'):
            # Format: g_{t}_{c}
            t = TL.index(int(parts[1]))
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
