"""
=========== Module Description ===========

This module provides functionality for constructing a complete binary tree structure, and creating literals
for the SAT solver based on the tree structure and dataset size.
"""

def build_complete_tree(depth):
    """
    Construct a complete binary tree of a specified depth with feature and threshold values for branching nodes.

    Parameters:
    - depth (int): The depth of the tree, with the root node at depth 0.

    Returns:
    - tree_structure (list): A list where each element represents a node in the tree.
    - TB (list): The indices of the branching nodes within the tree list.
    - TL (list): The indices of the leaf nodes within the tree list.
    """
    num_nodes = (2 ** (depth + 1)) - 1
    tree_structure = [None] * num_nodes
    TB, TL = [], []

    for node in range(num_nodes):
        if node < ((2 ** depth) - 1):
            TB.append(node)
            # Include feature and threshold keys for branching nodes
            tree_structure[node] = {
                'type': 'branching',
                'children': [2 * node + 1, 2 * node + 2],
                'feature': None,
                'threshold': None
            }
        else:
            TL.append(node)
            tree_structure[node] = {'type': 'leaf', 'label': None}

    return tree_structure, TB, TL


def create_literals(TB, TL, F, C, dataset_size, fixed_tree=False):
    """
    Create the literals for the SAT solver based on the tree structure and dataset size.

    This function creates four types of literals:
    - 'a' literals for feature splits at branching nodes,
    - 's' literals for data points directed to left or right,
    - 'z' literals for data points that end up at a leaf node,
    - 'g' literals for assigning class labels to leaf nodes.

    Parameters:
    - TB (list): Indices of branching nodes in the tree.
    - TL (list): Indices of leaf nodes in the tree.
    - num_features (int): The number of features in the dataset.
    - labels (list): The list of class labels for the dataset.
    - dataset_size (int): The number of data points in the dataset.

    Returns:
    - literals (dict): A dictionary where keys are literal names and values are their corresponding indices for the SAT solver.
    """

    literals = {}
    current_index = 1

    # Create 'a' literals for feature splits at branching nodes
    for t in TB:
        for j in F:
            literals[f'a_{t}_{j}'] = current_index
            current_index += 1

    # Create 's' literals for data points directed left or right at branching nodes
    for i in range(dataset_size):
        for t in TB:
            literals[f's_{i}_{t}'] = current_index
            current_index += 1

    # Create 'z' literals for data points ending up at leaf nodes
    for i in range(dataset_size):
        for t in TL:
            literals[f'z_{i}_{t}'] = current_index
            current_index += 1

    # Create 'g' literals for labels at leaf nodes
    for t in TL:
        for c in C:
            literals[f'g_{t}_{c}'] = current_index
            current_index += 1

    if fixed_tree:
        # Create 'p' literals for checking correct label association given to training data point
        for i in range(dataset_size):
            literals[f'p_{i}'] = current_index
            current_index += 1

    return literals, current_index
