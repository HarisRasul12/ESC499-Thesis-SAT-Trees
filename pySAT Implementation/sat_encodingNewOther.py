### Made by Haris Rasul
### Oct 15th 2023
### Purpose: SAT Encodings
# sat_encoding.py
from pysat.formula import CNF
from pysat.solvers import Solver
from collections import defaultdict
from DecisionTree import * 

# Clauses for minimal height optimal Decision:

# A dictionary to store the mapping between (prefix, t, j) tuples and unique variable indices
var_mapping = {}
current_index = 1
node_mapping = defaultdict(int)


# Placeholder functions for A_l(t) and A_r(t). Simple code for  tracking anestr nodes 
# Implement A_l(t) and A_r(t) based on decision tree structure.
def A_l(t):
    ancestors = []
    while t in node_mapping:
        parent = node_mapping[t]
        if t == parent * 2:  # child node has index 2* node of anecstr 
            ancestors.append(parent)
        t = parent
    return ancestors

def A_r(t):
    ancestors = []
    while t in node_mapping:
        parent = node_mapping[t]
        if t == parent * 2 + 1:  # A right child has index parent * 2 + 1
            ancestors.append(parent)
        t = parent
    return ancestors

def build_node_mapping(TB):
    for t in TB:
        node_mapping[t*2] = t  # Left child
        node_mapping[t*2 + 1] = t  # Right child


def choose_one_feature_at_branching(TB, F):
    """
    Encode that exactly one feature is chosen at each branching node.

    Parameters:
    - TB (list): List of branching nodes.
    - F (list): List of features.

    Returns:
    - CNF: A CNF formula encoding the decision tree logic for choosing one feature at branching.
    """
    
    formula = CNF()

    # Eq. (1): Ensuring that two different features aren't chosen for the same branching node.
    for t in TB:
        for j in F:
            for j_prime in F:
                if j != j_prime:
                    formula.append([-1 * get_var_index('a', t, j), -1 * get_var_index('a', t, j_prime)])
    
    # Eq. (2): Guaranteeing that at least one feature is chosen for every branching node.
    for t in TB:
        clause = [get_var_index('a', t, j) for j in F]
        formula.append(clause)
    
    print("Clauses from choose_one_feature_at_branching:")
    for clause in formula.clauses:
        print(clause)

    return formula

# all contsraint variable information on nodes 
def get_var_index(prefix, t, j_or_i):
    """
    Helper function to get the variable index for the SAT solver.

    Parameters:
    - prefix (str): Prefix of the variable type (e.g., 'a' for at,j or 'z' for zi,t).
    - t (int): Index of the branching node or leaf node.
    - j_or_i (int): Index of the feature or data point.

    Returns:
    - int: Unique variable index.
    """
    global current_index
    key = (prefix, t, j_or_i)
    
    # If the key hasn't been seen before, assign it a new unique index
    if key not in var_mapping:
        var_mapping[key] = current_index
        current_index += 1
    
    return var_mapping[key]

# {O_ij} list for feature ordering 
def compute_ordering(X, j):
    """
    Compute the ordering Oj for feature j based on dataset X.
    
    Parameters:
    - X (list): List of data points.
    - j (int): Index of the feature.

    Returns:
    - list: A list of consecutive pairs in the ordering.
    """
    sorted_indices = sorted(range(len(X)), key=lambda i: X[i][j])
    return [(sorted_indices[i], sorted_indices[i+1]) for i in range(len(X)-1)]

# O_ij clause
def enforce_ordering_at_branching(TB, F, X):
    """
    Encode the direction of data points based on feature values.

    Parameters:
    - TB (list): List of branching nodes.
    - F (list): List of features.
    - X (list): List of data points.

    Returns:
    - CNF: A CNF formula encoding the decision tree logic.
    """
    formula = CNF()
    
    for j in F:
        Oj = compute_ordering(X, j)
        for t in TB:
            for (i, i_prime) in Oj:
                # Clause (3)
                formula.append([-1 * get_var_index('a', t, j), get_var_index('s', i, t), -1 * get_var_index('s', i_prime, t)])
                # Clause (4)
                if X[i][j] == X[i_prime][j]:
                    formula.append([-1 * get_var_index('a', t, j), -1 * get_var_index('s', i, t), get_var_index('s', i_prime, t)])


    print("Clauses from enforce_ordering_at_branching:")
    for clause in formula.clauses:
        print(clause)

    return formula


def enforce_path_validity(TL, X):
    """
    Encode the path validity for each data point and leaf node.

    Parameters:
    - TL (list): List of leaf nodes.
    - X (list): List of data points.

    Returns:
    - CNF: A CNF formula encoding the decision tree path validity logic.
    """
    formula = CNF()

    for t in TL:
        for xi_index, xi in enumerate(X, start=1):  # start=1 to match the gamma indices
            for t_prime in A_l(t):
                # Eq. (5)
                formula.append([-1 * get_var_index('z', xi_index, t), get_var_index('s', xi_index, t_prime)])
            
            for t_prime in A_r(t):
                # Eq. (6)
                formula.append([-1 * get_var_index('z', xi_index, t), -1 * get_var_index('s', xi_index, t_prime)])
        
        # Eq. (7)
        left_clauses = [-1 * get_var_index('s', xi_index, t_prime) for t_prime in A_l(t)]
        right_clauses = [get_var_index('s', xi_index, t_prime) for t_prime in A_r(t)]
        
        formula.append([get_var_index('z', xi_index, t)] + left_clauses + right_clauses)
    
    print("Clauses from enforce_path_validity:")
    for clause in formula.clauses:
        print(clause)

    return formula

# Hopefully my ancestor node detection functions A(L) and A(r) are correct

def enforce_leaf_node_labels(TL, C):
    """
    Enforce Clause (8): Ensure that each leaf node has at most one label.
    This clause ensures that every leaf node is assigned at most one label. 
    Essentially, it makes sure that a single leaf does not get conflicting labels.

    Parameters:
    - TL (list): List of leaf nodes.
    - C (list): List of class labels.

    Returns:
    - CNF: A CNF formula encoding the constraint.
    """
    formula = CNF()
    for t in TL:
        for c in C:
            for c_prime in C:
                if c != c_prime:
                    formula.append([-1 * get_var_index('g', t, c), -1 * get_var_index('g', t, c_prime)])
    
    
    print("Clauses from enforce_leaf_node_labels:")
    for clause in formula.clauses:
        print(clause)

    return formula

def enforce_redundant_constraints(TB, F, X):
    """
    Enforce Clause (9) & (10): Redundant constraints to prune the search space.
    - Clause (9): For any branching node with a chosen feature `j`, the data point 
      with the smallest value for feature `j` is directed left.
    - Clause (10): For any branching node with a chosen feature `j`, the data point 
      with the largest value for feature `j` is directed right.

    Parameters:
    - TB (list): List of branching nodes.
    - F (list): List of features.
    - X (list): List of data points.

    Returns:
    - CNF: A CNF formula encoding the constraints.
    """
    formula = CNF()
    for t in TB:
        for j in F:
            # Data point with the lowest feature value is directed left
            formula.append([-1 * get_var_index('a', t, j), get_var_index('s', 1, t)])
            # Data point with the highest feature value is directed right
            formula.append([-1 * get_var_index('a', t, j), -1 * get_var_index('s', len(X), t)])
    
    
    print("Clauses from enforce_redundant_constraints:")
    for clause in formula.clauses:
        print(clause)

    return formula


def enforce_minimal_height_optimal_decision(TL, X, gamma):
    """
    Encode the minimal height optimal decision tree clauses.
    Parameters:
    - TL (list): List of leaf nodes.
    - X (list): List of data points.
    - gamma (dict): Training labels for data points. Keys are indices of data points and values are labels.

    Returns:
    - CNF: A CNF formula encoding the decision tree logic for minimal height.
    """
    formula = CNF()

    for xi_index, xi in enumerate(X, start=1):  # start=1 to match the gamma indices
        for t in TL:
            # Eq. (11)
            negated_clause = [-1 * get_var_index('z', xi_index, t), get_var_index('g', t, gamma[xi_index])]
            formula.append(negated_clause)

    print("Clauses from enforce_minimal_height_optimal_decision:")
    for clause in formula.clauses:
        print(clause)

    return formula


def create_cnf_for_decision_tree(TB, TL, F, X, C, gamma):
    cnf = CNF()
    
    # Add clauses from each function
    cnf.extend(choose_one_feature_at_branching(TB, F))
    cnf.extend(enforce_ordering_at_branching(TB, F, X))
    cnf.extend(enforce_path_validity(TL, X))
    cnf.extend(enforce_leaf_node_labels(TL, C))
    cnf.extend(enforce_redundant_constraints(TB, F, X))
    cnf.extend(enforce_minimal_height_optimal_decision(TL, X, gamma))
    

    #print("Complete CNF Clauses:")
    #for clause in cnf.clauses:
    #    print(clause)
    
    return cnf

def generate_tree_structure(depth):
    TB = []  # Branching nodes
    TL = []  # Leaf nodes
    for i in range(1, 2 ** depth):
        if i < 2 ** (depth - 1):
            TB.append(i)
        else:
            TL.append(i)
    return TB, TL


def decode_solution_to_tree(model, TB, TL, F, X, gamma):
    # Create a dictionary to hold the TreeNode objects
    nodes = {}

    # Create all nodes first
    for t in TB + TL:
        is_leaf = t in TL
        nodes[t] = create_tree_node(t, is_leaf)

    # Set the properties of each node based on the model
    for t in TB:
        # Find the feature for which at,j is true
        for j in F:
            if model[var_mapping[('a', t, j)]] > 0:  # If the variable is true in the model
                nodes[t].feature = j
                # Find the threshold based on the ordering
                Oj = compute_ordering(X, j)
                for (i, i_prime) in Oj:
                    if model[var_mapping[('s', i, t)]] > 0 and model[var_mapping[('s', i_prime, t)]] < 0:
                        nodes[t].threshold = X[i][j]
                        break
                break

    for t in TL:
        # Find the class label for which gt,c is true
        for c in gamma.values():
            if model[var_mapping[('g', t, c)]] > 0:  # If the variable is true in the model
                nodes[t].label = c
                break

    # Link the nodes to form the tree
    for t in TB:
        left_child_id = t * 2
        right_child_id = t * 2 + 1
        if left_child_id in nodes:
            nodes[t].set_children(nodes[left_child_id], nodes[right_child_id])
            nodes[left_child_id].set_parent(nodes[t])
            nodes[right_child_id].set_parent(nodes[t])

    # Find the root node (which has no parent) and create the tree
    root = next(node for node in nodes.values() if node.parent is None)
    tree = DecisionTree(root)

    return tree

def verify_solution(tree, X, gamma):
    for xi_index, xi in enumerate(X, start=1):
        prediction = tree.predict(xi)
        if prediction != gamma[xi_index]:
            return False
    return True

def iterative_tree_solver(F, X, C, gamma):
    depth = 1
    solution = None
    while solution is None:
        TB, TL = generate_tree_structure(depth)
        build_node_mapping(TB)
        cnf = create_cnf_for_decision_tree(TB, TL, F, X, C, gamma)
        solver = Solver()
        solver.append_formula(cnf)
        if solver.solve():
            model = solver.get_model()
            solution = decode_solution_to_tree(model, TB, TL, F, X, gamma)
            # Verify if the solution is correct
            if verify_solution(solution, X, gamma):
                break
        depth += 1
    return solution, depth




if __name__ == "__main__":
    # features
    F = ['F1', 'F2']

    # Toy dataset
    X = [
        {'F1': 5, 'F2': 3},  # Data point 1
        {'F1': 7, 'F2': 4},  # Data point 2
        {'F1': 2, 'F2': 6},  # Data point 3
        {'F1': 3, 'F2': 5},  # Data point 4
    ]

    # Labels for the dataset
    gamma = {
        1: 1,  # Label for data point 1
        2: 1,  # Label for data point 2
        3: 2,  # Label for data point 3
        4: 3,  # Label for data point 4
    }

    # class labels
    C = [1, 2, 3]

    # Generate the decision tree
    tree, depth = iterative_tree_solver(F, X, C, gamma)


    # Generate the decision tree
    tree, depth = iterative_tree_solver(F, X, C, gamma)


    
    # Print final set of clauses
    print("\nFinal Set of Clauses:")
    TB, TL = generate_tree_structure(depth)
    cnf = create_cnf_for_decision_tree(TB, TL, F, X, C, gamma)
    for clause in cnf.clauses:
        print(clause)

    # Check predictions
    correct_predictions = 0
    incorrect_predictions = 0
    print("\nPredictions:")
    for xi_index, xi in enumerate(X, start=1):
        predicted_label = tree.predict(xi)
        actual_label = gamma[xi_index]
        if predicted_label == actual_label:
            correct_predictions += 1
            print(f"Data point {xi_index}: Correct Prediction (Predicted: {predicted_label}, Actual: {actual_label})")
        else:
            incorrect_predictions += 1
            print(f"Data point {xi_index}: Incorrect Prediction (Predicted: {predicted_label}, Actual: {actual_label})")

    accuracy = tree.calculate_accuracy(X, gamma) * 100  # Convert to percentage
    print(f"\nAccuracy of the decision tree: {accuracy:.2f}%")

    
    
    
    # Visualize the tree
    tree.visualize()

   