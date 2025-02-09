"""
=========== Module Description ===========

Module for computing the maximum accuracy of fixed-height trees with categorical features at a given depth.
"""

from pysat.formula import WCNF

from satree.classification.min_height_tree_categorical_module import add_thresholds_categorical
from satree.classification.fixed_height_tree_module import solve_wcnf
from satree.classification.min_height_tree_module import visualize_tree
from satree.treemodder.builder import build_complete_tree, create_literals
from satree.classification.classification_clauses import add_clauses_for_features_and_paths, add_classification_clauses, add_feature_selection_clauses_for_branching_nodes


def build_clauses_categorical_fixed(literals, dataset, branch_nodes, leaf_nodes, num_features, features_categorical, features_numerical, labels, true_labels):
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding.

    Args:
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        branch_nodes (list): Indices of branching nodes.
        leaf_nodes (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.
        features_categorical (list): Indices of categorical features.
        features_numerical (list): Indices of numerical features.
        labels (list): Possible class labels for the data points.
        true_labels (list): True labels for the data points

    Returns:
        WCNF: A WCNF object containing all the clauses with unit weighst for cost
    """
    wcnf = WCNF()
    wcnf = add_feature_selection_clauses_for_branching_nodes(wcnf, literals, branch_nodes, num_features)
    wcnf = add_clauses_for_features_and_paths(wcnf, literals, dataset, branch_nodes, leaf_nodes, num_features, features_categorical, features_numerical, labels)
    wcnf = add_classification_clauses(wcnf, literals, dataset, leaf_nodes, true_labels)

    return wcnf


def find_fixed_depth_tree_categorical(features, features_categorical, features_numerical, labels, true_labels_for_points, dataset, depth):
    """
    Finds a fixed-depth decision tree for a categorical classification problem using SAT encoding.

    This function builds a complete decision tree of the specified depth, generates SAT literals, and
    constructs CNF clauses using a fixed encoding that handles both categorical and numerical features.
    It then attempts to solve the resulting CNF with a SAT solver. If a solution is found, the tree is
    augmented with computed thresholds (or splits) for its branching nodes, and the tree is visualized by
    rendering an image. If no solution is found, an error message is printed and "No solution" is returned.

    Args:
        features (list): List of feature identifiers used for splitting in the decision tree.
        features_categorical (list): List of identifiers for categorical features.
        features_numerical (list): List of identifiers for numerical features.
        labels (list): List of possible class labels.
        true_labels_for_points (list): The true labels for each data point in the dataset.
        dataset (array-like): The dataset where each row represents a data point.
        depth (int): The fixed depth to be used for constructing the decision tree.

    Returns:
        tuple: A tuple containing:
            - tree_with_thresholds: The decision tree structure with thresholds assigned to branching nodes.
            - literals (dict): A dictionary mapping SAT literal names to their corresponding variable indices.
            - depth (int): The fixed depth of the decision tree.
            - solution (list or str): The solution from the SAT solver if one is found, or "No solution" if unsolvable.
            - cost (int or float): The cost associated with the SAT solution.
    """
    tree, TB, TL = build_complete_tree(depth)
    literals = create_literals(TB, TL, features, labels, len(dataset), True)[0]
    wcnf = build_clauses_categorical_fixed(literals, dataset, TB, TL, len(features), features_categorical, features_numerical, labels,true_labels_for_points)
    solution,cost = solve_wcnf(wcnf, literals, TL, tree, labels, features)
    
    if solution != "No solution exists":
        tree_with_thresholds = add_thresholds_categorical(tree, literals, solution, dataset, features_categorical)
        dot = visualize_tree(tree_with_thresholds)
        dot.render(f'images/fixed_height/binary_decision_tree_fixed_with_categorical_features_depth_{depth}', format='png', cleanup=True)
    else:
        print('could not find solution')
        return 'No solution'
    
    return tree_with_thresholds, literals, depth, solution, cost

# if __name__ == "__main__":
#     # Define the test dataset parameters

#     # Numpy implementations 
#     depth = 2
#     features = np.array(['0', '1'])
#     features_categorical = np.array(['0']) #FC
#     features_numerical = np.array(['1']) #FN
#     labels = np.array([1, 0])
#     true_labels_for_points = np.array([1, 1, 0, 1, 0, 0, 0])
#     dataset = np.array([['A', 0], ['B', 3], ['C', 2], ['A', 1], ['A', 2], ['C', 0], ['C', 3]])  # Dataset X

#     tree_with_thresholds, literals, depth, solution, cost = find_fixed_depth_tree_categorical(features, features_categorical, features_numerical, labels, true_labels_for_points, dataset, depth)
#     print("The cost of the solution is: ", cost)
#     print(literals)
#     print(solution)
#     print(tree_with_thresholds)

#     print(compute_ordering_with_categorical(dataset,0,features_categorical))


