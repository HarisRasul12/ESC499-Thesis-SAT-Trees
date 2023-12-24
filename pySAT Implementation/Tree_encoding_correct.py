# Created by: Haris Rasul
# Date: November 19th 2023
# Python script to build the complete tree and create literals
# for a given depth and dataset. This script will be structured to run the test when executed as the main program.

from pysat.formula import CNF
from pysat.solvers import Solver
from graphviz import Digraph
import numpy as np

# Define the function to build a complete tree of a given depth
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



# Define the function to create literals based on the tree structure
def create_literals(TB, TL, F, C, dataset_size):
    """
    Create the literals for the SAT solver based on the tree structure and dataset size.

    This function creates four types of literals:
    - 'a' literals for feature splits at branching nodes,
    - 's' literals for data points directed to left or right,
    - 'z' literals for data points that end up at a leaf node,
    - 'g' literals for assigning class labels to leaf nodes.

    Parameters:
    - TB (list): Indices of branching nodes in the tree.
    - TL (list): Indices of leaf nodes in the tree.
    - num_features (int): The number of features in the dataset.
    - labels (list): The list of class labels for the dataset.
    - dataset_size (int): The number of data points in the dataset.

    Returns:
    - literals (dict): A dictionary where keys are literal names and values are their corresponding indices for the SAT solver.
    """

    literals = {}
    current_index = 1

    # Create 'a' literals for feature splits at branching nodes
    for t in TB:
        for j in F:
            literals[f'a_{t}_{j}'] = current_index
            current_index += 1

    # Create 's' literals for data points directed left or right at branching nodes
    for i in range(dataset_size):
        for t in TB:
            literals[f's_{i}_{t}'] = current_index
            current_index += 1

    # Create 'z' literals for data points ending up at leaf nodes
    for i in range(dataset_size):
        for t in TL:
            literals[f'z_{i}_{t}'] = current_index
            current_index += 1

    # Create 'g' literals for labels at leaf nodes
    for t in TL:
        for c in C:
            literals[f'g_{t}_{c}'] = current_index
            current_index += 1

    return literals

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

def build_clauses(literals, X, TB, TL, num_features, labels,true_labels):
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
    
    # Clause (1) and (2): Feature selection at branching nodes
    for t in TB:
        # At least one feature is chosen (Clause 2)
        clause = [literals[f'a_{t}_{j}'] for j in range(num_features)]
        cnf.append(clause)
        
        # No two features are chosen (Clause 1)
        for j in range(num_features):
            for jp in range(j + 1, num_features):
                clause = [-literals[f'a_{t}_{j}'], -literals[f'a_{t}_{jp}']]
                cnf.append(clause)

    # Clause (3) and (4): Data point direction based on feature values
    for j in range(num_features):
        Oj = compute_ordering(X, j)
        for (i, ip) in Oj:
            if X[i][j] < X[ip][j]:  # Different feature values (Clause 3)
                for t in TB:
                    cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i}_{t}'], -literals[f's_{ip}_{t}']])
            if X[i][j] == X[ip][j]:  # Equal feature values (Clause 4)
                for t in TB:
                    cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i}_{t}'], -literals[f's_{ip}_{t}']])
                    cnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{i}_{t}'], literals[f's_{ip}_{t}']])

    # Clause (5 and 6): Path valididty form right traversla and left traversal 
    for t in TL:
        left_ancestors = get_ancestors(t, 'left')
        right_ancestors = get_ancestors(t, 'right')
        # print('node: ', t)
        # print('left ancestors: ',left_ancestors)
        # print('right ancestors: ',right_ancestors)
        for i in range(len(X)):
            # Data point i ends at leaf node t (Clause 5 and 6) - assumption made!!!
            if left_ancestors:
                cnf.append([-literals[f'z_{i}_{t}']] + [literals[f's_{i}_{a}'] for a in left_ancestors])
            if right_ancestors:
                cnf.append([-literals[f'z_{i}_{t}']] + [-literals[f's_{i}_{a}'] for a in right_ancestors])

    # Clause (7): Each data point that does not end up in leaf node t has at least one deviation from the path
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

    # Clause (8): Each leaf node is assigned at most one label
    for t in TL:
        for c in range(len(labels)):
            for cp in range(c + 1, len(labels)):
                cnf.append([-literals[f'g_{t}_{labels[c]}'], -literals[f'g_{t}_{labels[cp]}']])

    # Clause (9) and (10): Redundant constraints to prune the search space
    # These clauses are optimizations
    for t in TB:
        # Find the data point with the lowest and highest feature value for each feature
        for j in range(num_features):
            sorted_by_feature = sorted(range(len(X)), key=lambda k: X[k][j])
            lowest_value_index = sorted_by_feature[0]
            highest_value_index = sorted_by_feature[-1]

            # Clause (9): The data point with the lowest feature value is directed left
            cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{lowest_value_index}_{t}']])

            # Clause (10): The data point with the highest feature value is directed right
            cnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{highest_value_index}_{t}']])

    # Clause (11): Correct class labels for leaf nodes
    for t in TL:
        for i, xi in enumerate(X):
            label = true_labels[i]
            cnf.append([-literals[f'z_{i}_{t}'], literals[f'g_{t}_{label}']])
    
    return cnf

def set_branch_node_features(model, literals, tree_structure,features,datasetX):
    """
    Set the chosen feature and threshold for each branching node in the tree structure
    based on the given SAT model.

    Args:
    - model (list): The model returned by the SAT solver.
    - literals (dict): A dictionary mapping literals to variable indices.
    - tree_structure (list): The complete binary tree structure.
    - dataset (list): The dataset, a list of tuples representing data points.
    - features (list): List of features in the dataset.
    """
    # For each branching node, determine the chosen feature and threshold
    for node_index in range(len(tree_structure)):
        #print(node_index)
        node = tree_structure[node_index]
        if node['type'] == 'branching':
            # Find which feature is used for splitting at the current node
            chosen_feature = None
            for feature in features:
                if literals[f'a_{node_index}_{feature}'] in model:
                    chosen_feature = feature
                    break
            
            # If a feature is chosen, set the feature and find the threshold
            if chosen_feature is not None:
                # Set the chosen feature and computed threshold in the tree structure
                node['feature'] = chosen_feature


def solve_cnf(cnf, literals, TL, tree_structure, labels,features,datasetX):
    """
    Attempts to solve the given CNF using a SAT solver.

    If a solution is found, it updates the tree structure with the correct labels for leaf nodes.

    Args:
    - cnf (CNF): The CNF object containing all clauses for the SAT solver.
    - literals (dict): A dictionary mapping literals to variable indices.
    - TL (list): Indices of leaf nodes in the tree.
    - tree_structure (list): The complete binary tree structure.
    - labels (list): The list of class labels for the dataset.

    Returns:
    - solution (list or str): The solution to the SAT problem if it exists, otherwise "No solution exists".
    """
    solver = Solver()
    solver.append_formula(cnf)
    if solver.solve():
        model = solver.get_model()
        # Update the tree structure with the correct labels for leaf nodes
        for t in TL:
            for label in labels:
                if literals[f'g_{t}_{label}'] in model:
                    tree_structure[t]['label'] = label
                    break
         # Set details for branching nodes
        set_branch_node_features(model, literals, tree_structure,features,datasetX)
        return model
    else:
        #print("no solution!")
        return "No solution exists"


# def add_thresholds(tree_structure, literals, model_solution, dataset):
#     """
#     Attempts to get the threhold on every branch node from tree structure.

#     If a solution is found, it updates the tree structure with the correct labels for leaf nodes.

#     Args:
#     - tree structure (list): tree structure that has barnch node, feature, leaf node, and labels detail
#     - liteals (dict): mapping index of liteal # to variabel
#     - model_solution (list): CNF solution with lireal to map back to variables
#     - dataset (list): list of tupes of features form my dataset ecah datapoint is a tuple 


#     Returns:
#     - tree_structure: final solution that is correct with trhehodlds appended to ecah branch node of the trees 
#     """
#     def get_literal_value(literal):
#         return literals[literal] if literals[literal] in model_solution else -literals[literal]

#     # Recursive function to set the threshold for each branching node
#     def set_thresholds(node_index, dataset_indices):
#         node = tree_structure[node_index]
#         if node['type'] == 'branching':
#             feature_index = int(node['feature'])
#             feature_values = [dataset[i][feature_index] for i in dataset_indices]
            
#             # Check if all feature values are the same
#             if len(set(feature_values)) == 1:
#                 # If all feature values are the same, set the threshold to the common feature value
#                 node['threshold'] = feature_values[0]
#             else:
#                 # Sort dataset indices by the feature value
#                 dataset_indices.sort(key=lambda i: dataset[i][feature_index])
#                 # Find the split point
#                 for i in range(1, len(dataset_indices)):
#                     left_index = dataset_indices[i - 1]
#                     right_index = dataset_indices[i]
#                     # Check if there's a change in direction between two consecutive data points
#                     if get_literal_value(f's_{left_index}_{node_index}') > 0 and get_literal_value(f's_{right_index}_{node_index}') < 0:
#                         # Threshold is the average value of the feature at the split point
#                         threshold = (dataset[left_index][feature_index] + dataset[right_index][feature_index]) / 2
#                         node['threshold'] = threshold
#                         break  # No need to check further once the split point is found
            
#             # Continue for children nodes
#             left_child_index, right_child_index = node['children']
#             # Left dataset indices are those for which the literal is positive
#             left_dataset_indices = [i for i in dataset_indices if get_literal_value(f's_{i}_{node_index}') > 0]
#             # Right dataset indices are the rest
#             right_dataset_indices = [i for i in dataset_indices if i not in left_dataset_indices]
#             # Recursively set thresholds for children nodes
#             set_thresholds(left_child_index, left_dataset_indices)
#             set_thresholds(right_child_index, right_dataset_indices)

#     # Start from the root node with all dataset indices
#     set_thresholds(0, list(range(len(dataset))))
#     return tree_structure

#adjusted Logic to compute threshold on the entire dataset at each feature node branch 
def add_thresholds(tree_structure, literals, model_solution, dataset):
    """
    Compute the threshold for each branching node in the tree structure
    based on the entire dataset.

    Args:
    - tree_structure (list): The binary tree structure containing nodes.
    - literals (dict): The mapping of literals to variable indices.
    - model_solution (list): The model solution from the SAT solver.
    - dataset (np.array): The dataset containing all the data points.

    Returns:
    - tree_structure (list): The updated tree structure with thresholds set for branching nodes.
    """
    def get_literal_value(literal):
        # Helper function to get the value of a literal from the model solution.
        return literals[literal] if literals[literal] in model_solution else -literals[literal]

    def set_thresholds(node_index, dataset):
        node = tree_structure[node_index]
        if node['type'] == 'branching':
            feature_index = int(node['feature'])

            # Instead of using the dataset_indices, we will compute the threshold based on the entire dataset.
            feature_values = dataset[:, feature_index]
            sorted_indices = np.argsort(feature_values)

            # Initialize the threshold
            threshold = None

            # Find the first instance where the direction changes and set the threshold.
            for i in range(1, len(sorted_indices)):
                left_index = sorted_indices[i - 1]
                right_index = sorted_indices[i]
                if get_literal_value(f's_{left_index}_{node_index}') > 0 and get_literal_value(f's_{right_index}_{node_index}') < 0:
                    threshold = (feature_values[left_index] + feature_values[right_index]) / 2
                    break

            # If no change in direction is found, threshold remains None.
            node['threshold'] = threshold
            
            # Continue for children nodes
            left_child_index, right_child_index = node['children'][0], node['children'][1]
            if left_child_index < len(tree_structure): # Check index is within bounds
                set_thresholds(left_child_index, dataset)
            if right_child_index < len(tree_structure): # Check index is within bounds
                set_thresholds(right_child_index, dataset)

    # Apply the threshold setting function starting from the root node
    set_thresholds(0, dataset)

    return tree_structure

# Create a matrix for each type of variable
def create_solution_matrix(literals, solution, var_type):
    # Find the maximum index for this var_type
    max_index = max(int(key.split('_')[1]) for key, value in literals.items() if key.startswith(var_type)) + 1
    max_sub_index = max(int(key.split('_')[2]) for key, value in literals.items() if key.startswith(var_type)) + 1
    
    # Initialize the matrix with zeros
    matrix = [[0 for _ in range(max_sub_index)] for _ in range(max_index)]
    
    # Fill in the matrix with 1 where the literals are true according to the solution
    for key, value in literals.items():
        if key.startswith(var_type):
            index, sub_index = map(int, key.split('_')[1:])
            matrix[index][sub_index] = 1 if value in solution else 0

    return matrix

# visualization code
def add_nodes(dot, tree, node_index=0):
    node = tree[node_index]
    if node['type'] == 'branching':
        dot.node(str(node_index), label=f"BranchNode:\n{node_index}\nFeature:{node['feature']}\nThreshold:{node['threshold']}")
        for child_index in node['children']:
            add_nodes(dot, tree, child_index)
            dot.edge(str(node_index), str(child_index))
    elif node['type'] == 'leaf':
        dot.node(str(node_index), label=f"LeafNode:\n{node_index}\nLabel: {node['label']}")

#visualization code
def visualize_tree(tree_structure):
    dot = Digraph()
    add_nodes(dot, tree_structure)
    return dot


def find_min_depth_tree(features, labels, true_labels_for_points, dataset):
    depth = 1  # Start with a depth of 1
    solution = "No solution exists"
    tree_with_thresholds = None
    tree = None
    literals = None

    while solution == "No solution exists":
        tree, TB, TL = build_complete_tree(depth)
        literals = create_literals(TB, TL, features, labels, len(dataset))
        cnf = build_clauses(literals, dataset, TB, TL, len(features), labels, true_labels_for_points)
        solution = solve_cnf(cnf, literals, TL, tree, labels, features, dataset)
        
        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds(tree, literals, solution, dataset)
            dot = visualize_tree(tree_with_thresholds)
            dot.render(f'binary_decision_tree_depth_{depth}', format='png', cleanup=True)
        else:
            depth += 1  # Increase the depth and try again
    
    return tree_with_thresholds, literals, depth, solution




# Test cases
if __name__ == "__main__":
    # Define the test dataset parameters
    
    # Test case 1 provided by Pouya 
    #depth = 2
    #features = ['0', '1']
    #labels = [1,0]
    #true_labels_for_points = [1,0,0,0]
    #dataset = [(1,1), (3,3), (3,1), (1,3)]  # Dataset X


    # Test case 2
    #depth = 1
    #features = ['0','1']
    #labels = [0,1]
    #true_labels_for_points = [0,1]
    #dataset = [(1,1),(1,2)]

    # Test case 3
    # depth = 2
    # features = ['0', '1']
    # labels = [1,2,3,4]
    # true_labels_for_points = [1,2,3,4]
    # dataset = [(1,1), (3,3), (3,1), (1,3)]  # Dataset X

    # # Build the complete tree
    # tree, TB, TL = build_complete_tree(depth)

    # # Create literals based on the tree structure and dataset
    # literals = create_literals(TB, TL, features, labels, len(dataset))
   
    # print("\nLiterals Created:")
    # for literal, index in literals.items():
    #     print(f"{literal}: {index}")
    
 
    # cnf = build_clauses(literals, dataset, TB, TL, len(features), labels,true_labels_for_points)

    # # Print out all clauses for verification
    # #print("\nClauses Created:")
    # #for clause in cnf.clauses:
    # #    print(clause)
    # #print("problem: ")
    # #print(cnf.clauses)

    # # Call the SAT solver and print the solution
    # solution = solve_cnf(cnf, literals, TL, tree, labels,features,dataset)
    # print("\nSAT Solver Output:")
    # print(solution)
    
    # if solution != "No solution exists":
        
    #     # Generate and visualize the tree

    #     tree_with_thresholds = add_thresholds(tree, literals, solution, dataset)
        
    #     #for node in tree_with_thresholds:
    #     #    print(node)

    #     dot = visualize_tree(tree_with_thresholds)
    #     dot.render('binary_decision_tree', format='png', cleanup=True)

    #     # create the matrix of each variable:

    #     # Print out the matrix for each type of variable
    #     print("\nSolution of Literals")
    #     for var_type in ['a', 's', 'z', 'g']:
    #         matrix = create_solution_matrix(literals, solution, var_type)
    #         print(f"{var_type.upper()} Variables:")
    #         for row in matrix:
    #             print(' '.join(map(str, row)))
    #         print("\n")

    
    # Test case 1 provided by Pouya 
    # features = ['0', '1']
    # labels = [1,0]
    # true_labels_for_points = [1,0,0,0]
    # dataset = [(1,1), (3,3), (3,1), (1,3)]  # Dataset X

    # Numpy implementations 
    features = np.array(['0', '1'])
    labels = np.array([1, 0])
    true_labels_for_points = np.array([1, 0, 0, 0])
    dataset = np.array([[1, 1], [3, 3], [3, 1], [1, 3]])  # Dataset X
    
    # Test case 2
    # features = ['0','1']
    # labels = [0,1]
    # true_labels_for_points = [0,1]
    # dataset = [(1,1),(1,2)]

    # Numpy implementations 
    # features = np.array(['0', '1'])
    # labels = np.array([1, 0])
    # true_labels_for_points = np.array([0, 1])
    # dataset = np.array([[1, 1], [1,2]])  # Dataset X

    # Test case 3
    # features = ['0', '1']
    # labels = [1,2,3,4]
    # true_labels_for_points = [1,2,3,4]
    # dataset = [(1,1), (3,3), (3,1), (1,3)]  # Dataset X

    # Numpy Implemnetations 
    # features = np.array(['0', '1'])
    # labels = np.array([1, 2, 3, 4])
    # true_labels_for_points = np.array([1, 2, 3, 4])
    # dataset = np.array([[1, 1], [3,3], [3,1], [1,3]])  # Dataset X

    min_depth_tree, min_depth_literals, min_depth,solution = find_min_depth_tree(features, labels, true_labels_for_points, dataset)
    print("Minimum Depth Tree Structure:")
    for node in min_depth_tree:
        print(node)
    print(f"Found at depth: {min_depth}")
    print("Literals at Minimum Depth:")
    for literal, index in min_depth_literals.items():
        print(f"{literal}: {index}")
    
    
    print("\nSolution to SAT: ")
    print(solution)
    
    # Print out the matrix for each type of variable
    print("\nSolution of Literals")
    for var_type in ['a', 's', 'z', 'g']:
        matrix = create_solution_matrix(min_depth_literals, solution, var_type)
        print(f"{var_type.upper()} Variables:")
        for row in matrix:
            print(' '.join(map(str, row)))
        print("\n")





