import numpy as np


# Helper function to sort data points by feature and create O_j FOR CATGEORICAL
def compute_ordering_with_categorical(X, feature_index, features_categorical):
    # Determine if the current feature is categorical
    is_categorical = str(feature_index) in features_categorical

    if is_categorical:
        # Group identical categories together and maintain their index order
        unique_categories = np.unique(X[:, feature_index])
        ordering = sum((list(np.where(X[:, feature_index] == category)[0])
                        for category in unique_categories), [])
    else:
        # For numerical features, convert to float then sort by feature value
        numerical_values = X[:, feature_index].astype(float)
        ordering = np.argsort(numerical_values).tolist()

    return ordering


def get_ancestors(node_index, side):
    """
    Find all the ancestors of a given node in the tree on the specified side (left or right).

    Parameters:
    - tree_structure (list): The complete binary tree structure.
    - node_index (int): The index of the leaf node for which to find ancestors.
    - side (str): Side of the ancestors to find ('left' or 'right').

    Returns:
    - ancestors (list): A list of indices of the ancestors on the specified side.
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


# Helper function to sort data points by feature and create O_j
def compute_ordering(X, feature_index):
    sorted_indices = sorted(range(len(X)), key=lambda i: X[i][feature_index])
    return [(sorted_indices[i], sorted_indices[i + 1]) for i in range(len(sorted_indices) - 1)]


def compute_numerical_threshold(feature_values, node_index, get_literal_value):
    """
    Computes the threshold for a numerical feature based on when the literal direction changes.

    Parameters:
      - feature_values (np.array): An array of values for a particular feature.
      - node_index (int): The index of the current node.
      - get_literal_value (function): A function that accepts a literal (string) and returns its value
                                      in the model solution.

    Returns:
      - threshold (float or None): The computed threshold as the average of the two consecutive feature
                                   values where the sign change occurs. Returns None if no change is found.
    """
    sorted_indices = np.argsort(feature_values)
    threshold = None
    for i in range(1, len(sorted_indices)):
        left_index = sorted_indices[i - 1]
        right_index = sorted_indices[i]
        if get_literal_value(f's_{left_index}_{node_index}') > 0 and get_literal_value(f's_{right_index}_{node_index}') < 0:
            threshold = (feature_values[left_index] + feature_values[right_index]) / 2
            break
    return threshold
