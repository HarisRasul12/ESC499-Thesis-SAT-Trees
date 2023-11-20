# Created by: Haris Rasul
# Date: November 19th 2023
# Python script to build the complete tree and create literals
# for a given depth and dataset. This script will be structured to run the test when executed as the main program.

from pysat.formula import CNF
from pysat.solvers import Solver
from graphviz import Digraph

# Define the function to build a complete tree of a given depth
def build_complete_tree(depth):
    """
    Construct a complete binary tree of a specified depth.

    Each node in the tree is represented as a dictionary in a list. The root node is at index 0.
    Branching nodes have the 'type' key set to 'branching' and a 'children' key listing their children's indices.
    Leaf nodes have the 'type' key set to 'leaf'.

    Parameters:
    - depth (int): The depth of the tree, with the root node at depth 0.

    Returns:
    - tree_structure (list): A list where each element represents a node in the tree.
    - TB (list): The indices of the branching nodes within the tree list.
    - TL (list): The indices of the leaf nodes within the tree list.
    """

    # Calculate the total number of nodes in a complete binary tree of the given depth
    num_nodes = (2 ** (depth + 1)) - 1
    # Initialize tree structure with placeholders for each node
    tree_structure = [None] * num_nodes
    # Initialize the list of branching nodes and leaf nodes
    TB, TL = [], []
    # Fill in the tree structure and identify branching and leaf nodes
    for node in range(num_nodes):
        if node < ((2 ** depth) - 1):
            TB.append(node)
            tree_structure[node] = {'type': 'branching', 'children': [2 * node + 1, 2 * node + 2]}
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

    # # Clause (5) and (6): Path validity based on ancestors
    # for t in TL:
    #     left_ancestors = get_ancestors(t, 'left')
    #     right_ancestors = get_ancestors(t, 'right')
    #     for i in range(len(X)):
    #         # Data point i ends at leaf node t (Clause 5 and 6)
    #         cnf.append([-literals[f'z_{i}_{t}']] + [literals[f's_{i}_{a}'] for a in left_ancestors])
    #         cnf.append([-literals[f'z_{i}_{t}']] + [-literals[f's_{i}_{a}'] for a in right_ancestors])

    # # Clause (7): Each data point that does not end up in leaf node t has at least one deviation from the path
    # for xi in range(len(X)):
    #     for t in TL:
    #         deviations = []
    #         left_ancestors = get_ancestors(t, 'left')  # Get left ancestors using TB indices
    #         right_ancestors = get_ancestors(t, 'right')  # Get right ancestors using TB indices
    #         for ancestor in left_ancestors:
    #             deviations.append(-literals[f's_{xi}_{ancestor}'])
    #         for ancestor in right_ancestors:
    #             deviations.append(literals[f's_{xi}_{ancestor}'])
    #         cnf.append([-literals[f'z_{xi}_{t}']] + deviations)
    for t in TL:
        left_ancestors = get_ancestors(t, 'left')
        right_ancestors = get_ancestors(t, 'right')
        for i in range(len(X)):
            # Data point i ends at leaf node t (Clause 5 and 6)
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

def solve_cnf(cnf, literals, TL, tree_structure, labels):
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
        return model
    else:
        print("no solution!")
        return "No solution exists"

def add_nodes(dot, tree, node_index=0):
    node = tree[node_index]
    if node['type'] == 'branching':
        dot.node(str(node_index), label=f"Branch\n{node_index}")
        for child_index in node['children']:
            add_nodes(dot, tree, child_index)
            dot.edge(str(node_index), str(child_index))
    elif node['type'] == 'leaf':
        dot.node(str(node_index), label=f"Leaf\n{node_index}\nLabel: {node['label']}")

def visualize_tree(tree_structure):
    dot = Digraph()
    add_nodes(dot, tree_structure)
    return dot

# Test the functions with the provided dataset under the main block
if __name__ == "__main__":
    # Define the test dataset parameters
    
    # Test case 1 provided by pouya 
    depth = 2
    features = ['0', '1']
    labels = [1,0]
    true_labels_for_points = [1,0,0,0]
    dataset = [(1,1), (3,3), (3,1), (1,3)]  # Dataset X


    #Test case 2
    #depth = 1
    #features = ['0','1']
    #labels = [0,1]
    #true_labels_for_points = [0,1]
    #dataset = [(1,0),(1,2)]


    # Build the complete tree
    tree, TB, TL = build_complete_tree(depth)

    # Create literals based on the tree structure and dataset
    literals = create_literals(TB, TL, features, labels, len(dataset))
    #print(literals)

    # Print the tree structure and literals for verification
    #print("Complete Binary Tree Structure:")
    #print(tree)
    #print("\nBranching Nodes (TB):", TB)
    #print("Leaf Nodes (TL):", TL)

    #print("\nLiterals Created:")
    #for literal, index in literals.items():
    #    print(f"{literal}: {index}")
    
    # Testing the get_ancestors function for left and right ancestors of a leaf node
    #left_ancestors = get_ancestors(5, 'left')  # Should return the root for the first leaf
    #right_ancestors = get_ancestors(4, 'right')  # Should return the root for the second leaf

    #print(f"Left ancestors of leaf node {5}: {left_ancestors}")
    #print(f"Right ancestors of leaf node {4}: {right_ancestors}")


    #feature orderings Testing:
    # a = compute_ordering(dataset, 0)
    # b = compute_ordering(dataset,1)
    # print("feature 1 Ordering: ", a)
    # print("feature 2 Ordering: ", b)

    # Tetsing the clause builder
    #print(len(features))
    cnf = build_clauses(literals, dataset, TB, TL, len(features), labels,true_labels_for_points)

    # Print out all clauses for verification
    #print("\nClauses Created:")
    #for clause in cnf.clauses:
    #    print(clause)
    #print("problem: ")
    #print(cnf.clauses)

    # Call the SAT solver and print the solution
    solution = solve_cnf(cnf, literals, TL, tree, labels)
    print("\nSAT Solver Output:")
    print(solution)

    # If a solution was found, print the updated tree structure
    if solution != "No solution exists":
        print("\nUpdated Tree Structure with Labels:")
        for node in tree:
            print(node)
    print(tree)
    
    # Generate and visualize the tree
    dot = visualize_tree(tree)
    dot.render('binary_decision_tree', format='png', cleanup=True)