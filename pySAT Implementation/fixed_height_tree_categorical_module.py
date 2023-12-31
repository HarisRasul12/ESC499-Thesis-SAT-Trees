# Created by: Haris Rasul
# Date: December 28 2023
# Python script module for computing fixed height trees max accuacy given fixed depth  with categorical features 

from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from graphviz import Digraph
import numpy as np
from min_height_tree_categorical_module import *
from fixed_height_tree_module import create_literals_fixed_tree, solve_wcnf


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
    
    # Clause (14) and (15): Feature selection at branching nodes
    for t in TB:
        # At least one feature is chosen (Clause 15)
        clause = [literals[f'a_{t}_{j}'] for j in range(num_features)]
        wcnf.append(clause)
        
        # No two features are chosen (Clause 14)
        for j in range(num_features):
            for jp in range(j + 1, num_features):
                clause = [-literals[f'a_{t}_{j}'], -literals[f'a_{t}_{jp}']]
                wcnf.append(clause)

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
                        wcnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{i_index}_{t}'], literals[f's_{ip_index}_{t}']]) 
                        # Clause (18) combination 
                        wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i_index}_{t}'], -literals[f's_{ip_index}_{t}']]) 
                else:
                    # Clause (16) and (17) for numerical features
                    # Use float conversion for numerical feature comparison 
                    if float(X[i_index, j]) < float(X[ip_index, j]):
                        wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i_index}_{t}'], -literals[f's_{ip_index}_{t}']]) #(16)
                    # Clause (17) and (16) added whem checks for the values are the same
                    if float(X[i_index, j]) == float(X[ip_index, j]):
                        wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i_index}_{t}'], -literals[f's_{ip_index}_{t}']]) #(16)
                        wcnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{i_index}_{t}'], literals[f's_{ip_index}_{t}']]) #(17)
    
    # Clause (19 and 20): Path valididty form right traversla and left traversal 
    for t in TL:
        left_ancestors = get_ancestors(t, 'left')
        right_ancestors = get_ancestors(t, 'right')
        for i in range(len(X)):
            # Data point i ends at leaf node t (Clause 5 and 6) - assumption made!!!
            if left_ancestors:
                wcnf.append([-literals[f'z_{i}_{t}']] + [literals[f's_{i}_{a}'] for a in left_ancestors])
            if right_ancestors:
                wcnf.append([-literals[f'z_{i}_{t}']] + [-literals[f's_{i}_{a}'] for a in right_ancestors])

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
                wcnf.append([literals[f'z_{xi}_{t}']] + deviations)    

    # Clause (22): Each leaf node is assigned at most one label
    for t in TL:
        for c in range(len(labels)):
            for cp in range(c + 1, len(labels)):
                wcnf.append([-literals[f'g_{t}_{labels[c]}'], -literals[f'g_{t}_{labels[cp]}']])

    # Clause (23) and (24)
    for t in TB:
        for j in range(num_features):
            ordering = compute_ordering_with_categorical(X, j, features_categorical)
            # Clause (23): Ensure that some arbitrary category or numerical value is directed left
            if str(j) in features_categorical or str(j) in features_numerical:
                # Use the first index in the ordered list as the lowest value for categorical or numerical feature
                wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{ordering[0]}_{t}']])
            # Clause (24): Only for numerical features, ensure the data point with the highest value is directed right
            if str(j) in features_numerical:
                # Use the last index in the ordered list as the highest value for the numerical feature
                wcnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{ordering[-1]}_{t}']])

    # New Hard Clause (25') for ensuring pi is true only when xi ends up in a leaf node with the correct label, REMOVED (CLAUSE 11)
    for i, xi in enumerate(X):
        for t in TL:
            label = true_labels[i]
            # This adds the clause (¬pi ∨ ¬zi,t ∨ gt,γ(xi))
            wcnf.append([-literals[f'p_{i}'], -literals[f'z_{i}_{t}'], literals[f'g_{t}_{label}']])
            
    # Add the soft clauses (26') for each data point being correctly classified
    for i in range(len(X)):
        wcnf.append([literals[f'p_{i}']], weight=1)

    return wcnf

def find_fixed_depth_tree_categorical(features, features_categorical, features_numerical, labels, true_labels_for_points, dataset, depth):
    solution = "No solution exists"
    tree_with_thresholds = None
    tree = None
    literals = None
    cost = None

    tree, TB, TL = build_complete_tree(depth)
    literals = create_literals_fixed_tree(TB, TL, features, labels, len(dataset))
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


