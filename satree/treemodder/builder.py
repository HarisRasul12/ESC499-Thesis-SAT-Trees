"""
=========== Module Description ===========

This module provides core utilities for constructing complete binary tree structures and generating SAT literals
that represent decision tree components within the SATree framework. The module facilitates the encoding of
decision trees as SAT formulas by:

  - Building a complete binary tree of a specified depth, where each node is represented as a dictionary containing
    its type (branching or leaf), child indices (for branching nodes), and placeholders for features, thresholds, or labels.

  - Generating SAT literals that correspond to various decision tree elements such as:
      • Feature selection at branching nodes ('a' literals),
      • Data point routing decisions at branching nodes ('s' literals),
      • Data point-to-leaf assignments ('z' literals),
      • Label assignments at leaf nodes ('g' literals),
      • (Optionally) Correct label indicators for training data points ('p' literals) in fixed tree encodings.

These functions form the backbone of the SAT encoding process for decision tree classification, enabling the
translation of tree structure and decision logic into a formal SAT representation.
"""

from typing import List, Dict, Any, Tuple

import numpy as np


def build_complete_tree(depth: int) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    """
    Constructs a complete binary tree of a specified depth and returns the tree structure along with
    the indices of branching nodes (TB) and leaf nodes (TL).

    The complete tree is represented implicitly as an array of dictionaries. For each node:
      - Branching nodes (internal nodes) are assigned keys for 'type' (set to 'branching'), a list of
        'children' (computed based on the node's index), and placeholders for 'feature' and 'threshold'.
      - Leaf nodes are assigned a 'type' of 'leaf' with a placeholder for 'label'.

    This representation is critical for subsequent SAT encoding, as it delineates which nodes will serve
    as decision points and which will output the final classification.

    Args:
        depth: The depth of the tree (with the root at depth 0).

    Returns:
        A tuple containing:
          - tree_structure: A list of dictionaries representing each node in the complete tree.
          - TB: A list of indices corresponding to branching nodes.
          - TL: A list of indices corresponding to leaf nodes.
    """

    num_nodes = (2 ** (depth + 1)) - 1
    tree_structure = [{} for _ in range(num_nodes)]
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


def create_literals(branch_nodes: List[int],
                    leaf_nodes: List[int],
                    feature_indices: np.ndarray,
                    class_labels: List[Any],
                    dataset_size: int,
                    fixed_tree: bool = False) -> Tuple[Dict[str, int], int]:
    """
    Generates SAT literals corresponding to the decision tree structure and classification objectives.

    This function creates a set of Boolean variables (literals) that are used in the SAT encoding of decision trees.
    The literals are organized as follows:
      - 'a' literals: Represent the selection of a feature at each branching node.
      - 's' literals: Encode the routing decisions for each data point at each branching node, indicating whether
        the point goes left or right.
      - 'z' literals: Represent the assignment of data points to specific leaf nodes.
      - 'g' literals: Correspond to the label assignments at each leaf node.
      - (Optional) 'p' literals: Used in fixed tree encodings to check that data points receive the correct label
        as per the training data.

    The function assigns a unique integer index to each literal, ensuring a consistent mapping for the SAT solver.

    Args:
        branch_nodes: List of indices corresponding to branching nodes.
        leaf_nodes: List of indices corresponding to leaf nodes.
        feature_indices: An array of identifiers for the features used in the tree splits.
        class_labels: A collection of possible class labels.
        dataset_size: The total number of data points in the dataset.
        fixed_tree: Boolean flag indicating whether to generate additional 'p' literals for fixed tree encodings.

    Returns:
        A tuple containing:
          - literals: A dictionary mapping literal names (strings) to their unique integer indices.
          - current_index: The next available index after all literals have been assigned.
    """

    literals = {}
    current_index = 1

    # Create 'a' literals for feature splits at branching nodes
    for t in branch_nodes:
        for j in feature_indices:
            literals[f'a_{t}_{j}'] = current_index
            current_index += 1

    # Create 's' literals for data points directed left or right at branching nodes
    for i in range(dataset_size):
        for t in branch_nodes:
            literals[f's_{i}_{t}'] = current_index
            current_index += 1

    # Create 'z' literals for data points ending up at leaf nodes
    for i in range(dataset_size):
        for t in leaf_nodes:
            literals[f'z_{i}_{t}'] = current_index
            current_index += 1

    # Create 'g' literals for labels at leaf nodes
    for t in leaf_nodes:
        for c in class_labels:
            literals[f'g_{t}_{c}'] = current_index
            current_index += 1

    if fixed_tree:
        # Create 'p' literals for checking correct label association given to training data point
        for i in range(dataset_size):
            literals[f'p_{i}'] = current_index
            current_index += 1

    return literals, current_index
