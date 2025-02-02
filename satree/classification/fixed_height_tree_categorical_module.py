"""
=========== Module Description ===========

Module for computing the maximum accuracy of fixed-height trees with categorical features at a given depth.
"""

from pysat.formula import WCNF

from min_height_tree_categorical_module import add_thresholds_categorical
from satree.classification.classification_core import compute_ordering_with_categorical
from fixed_height_tree_module import solve_wcnf
from min_height_tree_module import get_ancestors, visualize_tree
from satree.treemodder.builder import build_complete_tree, create_literals
from classification_clauses import add_clauses_for_features_and_paths, add_classification_clauses, add_feature_selection_clauses_for_branching_nodes


def build_clauses_categorical_fixed(literals, X, TB, TL, num_features, features_categorical, features_numerical, labels,true_labels):
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding.

    Args:
        literals (dict): A dictionary mapping literals to variable indices.
        X (list): The dataset, a list of tuples representing data points.
        TB (list): Indices of branching nodes.
        TL (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.
        labels (list): Possible class labels for the data points.

    Returns:
        WCNF: A WCNF object containing all the clauses with unit weighst for cost
    """
    wcnf = WCNF()
    wcnf = add_feature_selection_clauses_for_branching_nodes(wcnf, literals, TB, num_features)
    wcnf = add_clauses_for_features_and_paths(wcnf, literals, X, TB, TL, num_features, features_categorical, features_numerical, labels)
    wcnf = add_classification_clauses(wcnf, literals, X, TL, true_labels)

    return wcnf

def find_fixed_depth_tree_categorical(features, features_categorical, features_numerical, labels, true_labels_for_points, dataset, depth):
    solution = "No solution exists"
    tree_with_thresholds = None
    tree = None
    literals = None
    cost = None

    tree, TB, TL = build_complete_tree(depth)
    literals = create_literals(TB, TL, features, labels, len(dataset), True)[0]
    wcnf = build_clauses_categorical_fixed(literals, dataset, TB, TL, len(features), features_categorical, features_numerical, labels,true_labels_for_points)
    solution,cost = solve_wcnf(wcnf, literals, TL, tree, labels, features, dataset)
    
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


