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

from typing import List, Callable, Optional

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
