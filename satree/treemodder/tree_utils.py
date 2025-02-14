"""
=========== Module Description ===========

This module provides essential utilities for manipulating binary tree structures within the SATree framework. It focuses
on extracting structural properties from a binary tree represented implicitly as an array. In particular, the module
offers functionality to compute the ancestors of a given node on a specified side (left or right), which is pivotal
for enforcing path-based constraints in the SAT encoding of decision trees.

Mathematically, the tree is modeled as a complete binary tree where each node’s parent is computed as (i - 1) // 2.
This representation supports the construction of SAT clauses that ensure data points follow a valid path from the
root to a leaf node. By selectively retrieving ancestors based on whether the node is a left or right child, the module
aids in defining routing constraints and maintaining consistency in the decision-making process modeled by
the SAT formulation.
"""

from typing import List


def get_ancestors(node_index: int, side: str) -> List[int]:
    """
    Returns the indices of the ancestors of a node in a binary tree based on its implicit array representation.

    The function traverses upward from the given node index and collects the indices of ancestors that are on
    the specified side. The binary tree is assumed to be represented in an array where, for any node at index i,
    its parent is at index (i - 1) // 2.

    Args:
        node_index: The index of the node whose ancestors are to be found.
        side: The side of the ancestors to collect ('left' or 'right'). For example, if 'left', only ancestors
                    where the current node is a left child are included.

    Returns:
        A list of ancestor indices on the specified side.
    """
    ancestors = []
    current_index = node_index
    while True:
        parent_index = (current_index - 1) // 2
        if parent_index < 0:
            break
        # Check if current node is a left or right child
        if (side == 'left' and current_index % 2 == 1) or (side == 'right' and current_index % 2 == 0):
            ancestors.append(parent_index)
        current_index = parent_index
    return ancestors
