"""
=========== Module Description ===========

Module to build the complete minimum depth tree and create literals. This module includes a modified decoded threshold
compared to the original paper and should be tested on test accuracy later.
"""

from graphviz import Digraph
from pysat.formula import CNF
from pysat.solvers import Solver

from satree.classification.classification_core import compute_numerical_threshold
from satree.treemodder.builder import build_complete_tree, create_literals
from satree.classification.classification_clauses import add_redundant_constraints, construct_maxsat_clauses


def build_clauses(literals, dataset, branch_nodes, leaf_nodes, num_features, labels, true_labels):
    """
    Constructs the clauses for the SAT solver based on the decision tree encoding.

    Args:
        literals (dict): A dictionary mapping literals to variable indices.
        dataset (list): The dataset, a list of tuples representing data points.
        branch_nodes (list): Indices of branching nodes.
        leaf_nodes (list): Indices of leaf nodes.
        num_features (int): Number of features in the dataset.
        labels (list): Possible class labels for the data points.
        true_labels (list): The true class labels for the data points.

    Returns:
        CNF: A CNF object containing all the clauses.
    """
    cnf = CNF()

    cnf = construct_maxsat_clauses(cnf, literals, dataset, branch_nodes, leaf_nodes, num_features, labels)

    cnf = add_redundant_constraints(cnf, literals, dataset, branch_nodes, num_features)

    # Clause (11): Correct class labels for leaf nodes
    for t in leaf_nodes:
        for i, xi in enumerate(dataset):
            label = true_labels[i]
            cnf.append([-literals[f'z_{i}_{t}'], literals[f'g_{t}_{label}']])
    
    return cnf

def set_branch_node_features(model, literals, tree_structure,features):
    """
    Set the chosen feature and threshold for each branching node in the tree structure
    based on the given SAT model.

    Args:
        model (list): The model returned by the SAT solver.
        literals (dict): A dictionary mapping literals to variable indices.
        tree_structure (list): The complete binary tree structure.
        features (list): List of features in the dataset.

    Returns:
        None
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


def solve_cnf(cnf, literals, leaf_nodes, tree_structure, labels, features):
    """
    Attempts to solve the given CNF using a SAT solver.

    If a solution is found, it updates the tree structure with the correct labels for leaf nodes.

    Args:
        cnf (CNF): The CNF object containing all clauses for the SAT solver.
        literals (dict): A dictionary mapping literals to variable indices.
        leaf_nodes (list): Indices of leaf nodes in the tree.
        tree_structure (list): The complete binary tree structure.
        labels (list): The list of class labels for the dataset.
        features (list): The list of feature names in the dataset.

    Returns:
        solution (list or str): The solution to the SAT problem if it exists, otherwise "No solution exists".
    """
    solver = Solver()
    solver.append_formula(cnf)
    if solver.solve():
        model = solver.get_model()
        # Update the tree structure with the correct labels for leaf nodes
        for t in leaf_nodes:
            for label in labels:
                if literals[f'g_{t}_{label}'] in model:
                    tree_structure[t]['label'] = label
                    break
         # Set details for branching nodes
        set_branch_node_features(model, literals, tree_structure,features)
        return model
    else:
        #print("no solution!")
        return "No solution exists"


#adjusted Logic to compute threshold on the entire dataset at each feature node branch
def add_thresholds(tree_structure, literals, model_solution, dataset):
    """
    Compute the threshold for each branching node in the tree structure
    based on the entire dataset.

    Args:
        tree_structure (list): The binary tree structure containing nodes.
        literals (dict): The mapping of literals to variable indices.
        model_solution (list): The model solution from the SAT solver.
        dataset (np.array): The dataset containing all the data points.

    Returns:
        tree_structure (list): The updated tree structure with thresholds set for branching nodes.
    """
    def get_literal_value(literal):
        return literals[literal] if literals[literal] in model_solution else -literals[literal]

    def set_thresholds(node_index, data):
        node = tree_structure[node_index]
        if node['type'] == 'branching':
            feature_index = int(node['feature'])
            feature_values = data[:, feature_index]
            # Use the helper function to compute the threshold.
            node['threshold'] = compute_numerical_threshold(feature_values, node_index, get_literal_value)

            # Continue for children nodes.
            left_child_index, right_child_index = node['children'][0], node['children'][1]
            if left_child_index < len(tree_structure):
                set_thresholds(left_child_index, data)
            if right_child_index < len(tree_structure):
                set_thresholds(right_child_index, data)

    set_thresholds(0, dataset)
    return tree_structure


def add_nodes(dot, tree, node_index=0):
    """
    Recursively adds nodes and edges to a Graphviz Digraph object based on the given tree structure.

    This function traverses the tree (represented as a list of nodes, where each node is a dictionary)
    starting from the specified node index, adding each node and its connecting edges to the provided
    Digraph object for visualization.

    Args:
        dot (Digraph): A Graphviz Digraph object used for visualizing the tree.
        tree (list): The complete tree structure (list of nodes).
        node_index (int, optional): The index of the current node to process (defaults to 0 for the root).

    Returns:
        None. The function updates the Digraph object in place.
    """
    node = tree[node_index]
    if node['type'] == 'branching':
        dot.node(str(node_index), label=f"BranchNode:\n{node_index}\nFeature:{node['feature']}\nThreshold:{node['threshold']}")
        for child_index in node['children']:
            add_nodes(dot, tree, child_index)
            dot.edge(str(node_index), str(child_index))
    elif node['type'] == 'leaf':
        dot.node(str(node_index), label=f"LeafNode:\n{node_index}\nLabel: {node['label']}")


def visualize_tree(tree_structure):
    """
    Visualizes the given tree structure using Graphviz.

    This function creates a Graphviz Digraph, adds all the nodes and edges based on the tree structure,
    and returns the resulting Digraph object for rendering or further manipulation.

    Args:
        tree_structure (list): The complete tree structure (list of nodes) to be visualized.

    Returns:
        Digraph: A Graphviz Digraph object representing the tree.
    """
    dot = Digraph()
    add_nodes(dot, tree_structure)
    return dot


def find_min_depth_tree(features, labels, true_labels_for_points, dataset):
    """
    Finds the minimum depth decision tree for a classification problem using SAT-based encoding.

    The function incrementally increases the depth of a complete binary tree until a satisfiable solution is found.
    For each depth, it:
      - Builds a complete tree.
      - Generates SAT literals.
      - Constructs the corresponding CNF clauses.
      - Attempts to solve the CNF with a SAT solver.
      - If a solution is found, it computes thresholds for the branching nodes and visualizes the tree.
      - If no solution is found, it increases the depth and tries again.

    Args:
        features (list): List of feature identifiers used for splitting in the decision tree.
        labels (list): List of possible class labels.
        true_labels_for_points (list): The true class labels for each data point.
        dataset (array-like): The dataset containing data points (each data point is represented as a tuple or array).

    Returns:
        tuple: A tuple containing:
            - tree_with_thresholds: The final decision tree structure with computed thresholds.
            - literals (dict): A dictionary mapping SAT literal names to their variable indices.
            - depth (int): The depth of the found decision tree.
            - solution (list or str): The SAT solver's solution if found, or "No solution exists" if unsolvable.
    """
    depth = 1  # Start with a depth of 1
    solution = "No solution exists"
    tree_with_thresholds = None
    literals = None

    while solution == "No solution exists":
        tree, TB, TL = build_complete_tree(depth)
        literals = create_literals(TB, TL, features, labels, len(dataset), False)[0]
        cnf = build_clauses(literals, dataset, TB, TL, len(features), labels, true_labels_for_points)
        solution = solve_cnf(cnf, literals, TL, tree, labels, features)
        
        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds(tree, literals, solution, dataset)
            dot = visualize_tree(tree_with_thresholds)
            dot.render(f'images/min_height/binary_decision_tree_min_depth_{depth}', format='png', cleanup=True)
        else:
            print('no solution at depth', depth)
            depth += 1  # Increase the depth and try again
    
    return tree_with_thresholds, literals, depth, solution

