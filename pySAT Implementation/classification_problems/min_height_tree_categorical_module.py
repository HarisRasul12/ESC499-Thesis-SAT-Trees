# Created by: Haris Rasul
# Date: December 27 2023
# Python script base to help solve SAT Problems Categorical features with numerical features

from pysat.formula import CNF
from pysat.solvers import Solver
from graphviz import Digraph
import numpy as np
from classification_problems.min_height_tree_module import get_ancestors, build_complete_tree, create_literals, solve_cnf, visualize_tree

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

# Caluses builder for dataeest with cateorgoialc features
def build_clauses_categorical(literals, X, TB, TL, num_features, features_categorical, features_numerical, labels,true_labels):
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
        CNF: A CNF object containing all the clauses.
    """
    cnf = CNF()
    
    # Clause (14) and (15): Feature selection at branching nodes
    for t in TB:
        # At least one feature is chosen (Clause 15)
        clause = [literals[f'a_{t}_{j}'] for j in range(num_features)]
        cnf.append(clause)
        
        # No two features are chosen (Clause 14)
        for j in range(num_features):
            for jp in range(j + 1, num_features):
                clause = [-literals[f'a_{t}_{j}'], -literals[f'a_{t}_{jp}']]
                cnf.append(clause)

    # Clauses (16), (17), and (18)
    for j in range(num_features):
        ordering = compute_ordering_with_categorical(X, j, features_categorical)
        for t in TB:
            for i in range(len(ordering) - 1):
                i_index, ip_index = ordering[i], ordering[i + 1]
                if str(j) in features_categorical:
                    # Clause (18) and (17) for categorical features 
                    # we add Eq. (18) that, together with the existing Eq. (17), 
                    # guarantees that data points with the same category are directed in the same direction. 
                    if X[i_index, j] == X[ip_index, j]:
                        # Clause (17)
                        cnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{i_index}_{t}'], literals[f's_{ip_index}_{t}']]) 
                        # Clause (18) combination 
                        cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i_index}_{t}'], -literals[f's_{ip_index}_{t}']]) 
                else:
                    # Clause (16) and (17) for numerical features
                    # Use float conversion for numerical feature comparison   
                    if float(X[i_index, j]) < float(X[ip_index, j]):
                        cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i_index}_{t}'], -literals[f's_{ip_index}_{t}']]) #(16)
                    # Clause (17) and (16) added whem checks for the values are the same
                    if float(X[i_index, j]) == float(X[ip_index, j]):
                        cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i_index}_{t}'], -literals[f's_{ip_index}_{t}']]) #(16)
                        cnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{i_index}_{t}'], literals[f's_{ip_index}_{t}']]) #(17)
    
    # Clause (19 and 20): Path valididty form right traversla and left traversal 
    for t in TL:
        left_ancestors = get_ancestors(t, 'left')
        right_ancestors = get_ancestors(t, 'right')
        for i in range(len(X)):
            # Data point i ends at leaf node t (Clause 5 and 6) - assumption made!!!
            if left_ancestors:
                cnf.append([-literals[f'z_{i}_{t}']] + [literals[f's_{i}_{a}'] for a in left_ancestors])
            if right_ancestors:
                cnf.append([-literals[f'z_{i}_{t}']] + [-literals[f's_{i}_{a}'] for a in right_ancestors])

    # Clause (21): Each data point that does not end up in leaf node t has at least one deviation from the path
    for xi in range(len(X)):
        for t in TL:
            deviations = []
            left_ancestors = get_ancestors(t, 'left')  # Get left ancestors using TB indices
            right_ancestors = get_ancestors(t, 'right')  # Get right ancestors using TB indices
            # Only append deviations if there are ancestors on the corresponding side
            if left_ancestors:
                deviations.extend([-literals[f's_{xi}_{ancestor}'] for ancestor in left_ancestors])
            if right_ancestors:
                deviations.extend([literals[f's_{xi}_{ancestor}'] for ancestor in right_ancestors])
            # Only append the clause if there are any deviations
            if deviations:
                cnf.append([literals[f'z_{xi}_{t}']] + deviations)    

    # Clause (22): Each leaf node is assigned at most one label
    for t in TL:
        for c in range(len(labels)):
            for cp in range(c + 1, len(labels)):
                cnf.append([-literals[f'g_{t}_{labels[c]}'], -literals[f'g_{t}_{labels[cp]}']])

    # Clause (23) and (24)
    for t in TB:
        for j in range(num_features):
            ordering = compute_ordering_with_categorical(X, j, features_categorical)
            # Clause (23): Ensure that some arbitrary category or numerical value is directed left
            if str(j) in features_categorical or str(j) in features_numerical:
                # Use the first index in the ordered list as the lowest value for categorical or numerical feature
                cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{ordering[0]}_{t}']])
            # Clause (24): Only for numerical features, ensure the data point with the highest value is directed right
            if str(j) in features_numerical:
                # Use the last index in the ordered list as the highest value for the numerical feature
                cnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{ordering[-1]}_{t}']])

    # Clause (25): Correct class labels for leaf nodes
    for t in TL:
        for i, xi in enumerate(X):
            label = true_labels[i]
            cnf.append([-literals[f'z_{i}_{t}'], literals[f'g_{t}_{label}']])
    
    return cnf

#adjusted Logic to compute threshold on the entire dataset at each feature node branch and categorical feature 
def add_thresholds_categorical(tree_structure, literals, model_solution, dataset, features_categorical):
    def get_literal_value(literal):
        return literals[literal] if literals[literal] in model_solution else -literals[literal]

    def set_thresholds_categorical(node_index, dataset):
        node = tree_structure[node_index]
        if node['type'] == 'branching':
            feature_index = int(node['feature'])
            is_categorical = str(feature_index) in features_categorical
            
            if is_categorical:
                # For categorical features, list the unique values that went left
                categories_that_went_left = set()
                for i, data_point in enumerate(dataset):
                    if get_literal_value(f's_{i}_{node_index}') > 0:
                        categories_that_went_left.add(data_point[feature_index])
                node['threshold'] = sorted(list(categories_that_went_left))
            else:
                # For numerical features, use the existing logic to find the threshold
                feature_values = dataset[:, feature_index].astype(float)
                sorted_indices = np.argsort(feature_values)
                threshold = None
                for i in range(1, len(sorted_indices)):
                    left_index = sorted_indices[i - 1]
                    right_index = sorted_indices[i]
                    if get_literal_value(f's_{left_index}_{node_index}') > 0 and get_literal_value(f's_{right_index}_{node_index}') < 0:
                        threshold = (feature_values[left_index] + feature_values[right_index]) / 2
                        break
                node['threshold'] = threshold
            
            # Continue for children nodes
            left_child_index, right_child_index = node['children'][0], node['children'][1]
            if left_child_index < len(tree_structure):
                set_thresholds_categorical(left_child_index, dataset)
            if right_child_index < len(tree_structure):
                set_thresholds_categorical(right_child_index, dataset)

    # Apply the threshold setting function starting from the root node
    set_thresholds_categorical(0, dataset)
    return tree_structure

def find_min_depth_tree_categorical(features, features_categorical, features_numerical, labels, true_labels_for_points, dataset):
    depth = 1  # Start with a depth of 1
    solution = "No solution exists"
    tree_with_thresholds = None
    tree = None
    literals = None

    while solution == "No solution exists":
        tree, TB, TL = build_complete_tree(depth)
        literals = create_literals(TB, TL, features, labels, len(dataset))
        cnf = build_clauses_categorical(literals, dataset, TB, TL, len(features), features_categorical, features_numerical, labels, true_labels_for_points)
        solution = solve_cnf(cnf, literals, TL, tree, labels, features, dataset)
        
        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds_categorical(tree, literals, solution, dataset, features_categorical)
            dot = visualize_tree(tree_with_thresholds)
            dot.render(f'images/min_height/binary_decision_tree_min_depth_with_categorical_features_depth_{depth}', format='png', cleanup=True)
        else:
            print("No solution at depth: ", depth)
            depth += 1  # Increase the depth and try again
    
    return tree_with_thresholds, literals, depth, solution
