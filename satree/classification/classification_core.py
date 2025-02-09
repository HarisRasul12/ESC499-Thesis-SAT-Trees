"""
=========== Module Description ===========

This module provides core utility functions for encoding and analyzing decision tree
classification problems using SAT-based methods. It includes functions to:

  - Compute an ordering of data points for a given feature. For categorical features,
    indices are grouped by unique category (preserving the natural order), while for
    numerical features, the indices are sorted by their numeric values.

  - Determine the ancestors of a node in a binary tree represented implicitly as an array,
    which is useful for constructing path-based constraints.

  - Generate consecutive pairs of data point indices based on sorted order for a numerical
    feature. These pairs are used to establish ordering constraints.

  - Compute a numerical threshold for a feature at a decision node by detecting the point
    (between two consecutive data points) where the SAT literal assignment changes sign. This
    threshold represents a decision boundary in the tree.

Together, these utilities support the construction and analysis of SAT encodings for decision
trees, enabling the formulation of constraints that ensure proper data point routing, feature
selection, and threshold determination for both categorical and numerical data.
"""

from typing import List, Tuple, Callable, Optional

import numpy as np


def compute_ordering_with_categorical(dataset: np.ndarray,
                                      feature_index: int,
                                      features_categorical: List[str]) -> List[int]:
    """
    Computes an ordering of data point indices for a given feature, taking into account categorical features.

    For a categorical feature, the function groups data points by unique category and concatenates the indices
    within each group (preserving their natural order). For a numerical feature, it returns the sorted indices
    based on the float value of the feature.

    Args:
        dataset: A 2D array representing the dataset, where each row is a data point.
        feature_index: The index of the feature to order by.
        features_categorical: A list of feature indices (as strings) that are considered categorical.

    Returns:
        An ordered list of data point indices. For categorical features, indices are grouped by category;
                    for numerical features, indices are sorted by value.
    """
    # Determine if the current feature is categorical
    is_categorical = str(feature_index) in features_categorical

    if is_categorical:
        # Group identical categories together and maintain their index order
        unique_categories = np.unique(dataset[:, feature_index])
        ordering = sum((list(np.where(dataset[:, feature_index] == category)[0])
                        for category in unique_categories), [])
    else:
        # For numerical features, convert to float then sort by feature value
        numerical_values = dataset[:, feature_index].astype(float)
        ordering = np.argsort(numerical_values).tolist()

    return ordering


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


def compute_ordering(dataset: np.ndarray,
                     feature_index: int) -> List[Tuple[int, int]]:
    """
    Computes the ordering of data point indices for a specified feature.

    The function sorts the data points based on the value of the specified feature and returns a list of tuples,
    each containing a pair of consecutive data point indices from the sorted order.

    Args:
        dataset: A list of data points (each data point can be a tuple or list).
        feature_index: The index of the feature used for sorting.

    Returns:
        A list of tuples, where each tuple contains a pair (i, j) representing consecutive data point indices
                    in the sorted order.
    """
    sorted_indices = sorted(range(len(dataset)), key=lambda i: float(dataset[i][feature_index]))
    return [(sorted_indices[i], sorted_indices[i + 1]) for i in range(len(sorted_indices) - 1)]


def compute_numerical_threshold(feature_values: np.ndarray,
                                node_index: int,
                                get_literal_value: Callable[[str], int]) -> Optional[float]:
    """
    Computes the threshold for a numerical feature based on when the literal direction changes.

    The threshold is computed as the average of two consecutive feature values where the direction (sign)
    of the corresponding literals changes. If no such change is found, returns None.

    Args:
        feature_values: An array of values for a particular feature.
        node_index: The index of the current node.
        get_literal_value: A function that accepts a literal (str) and returns its value in the model solution.

    Returns:
        The computed threshold as the average of the two adjacent feature values where the sign
                       change occurs, or None if no change is found.
    """
    sorted_indices = np.argsort(feature_values)
    threshold = None
    for i in range(1, len(sorted_indices)):
        left_index = sorted_indices[i - 1]
        right_index = sorted_indices[i]
        if get_literal_value(f's_{left_index}_{node_index}') > 0 > get_literal_value(f's_{right_index}_{node_index}'):
            threshold = (feature_values[left_index] + feature_values[right_index]) / 2
            break
    return threshold
