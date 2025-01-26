"""
=========== Module Description ===========

This module provides functionality for constructing a complete binary tree structure.
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
