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
#current_index = 1
#var_mapping = {}
#node_mapping = defaultdict(int)


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
        for f in F:
            # Find the data point with the lowest and highest value for feature f
            sorted_by_feature = sorted(X, key=lambda k: k[f])
            i_min = sorted_by_feature[0]  # Get the data point with the lowest feature value
            i_max = sorted_by_feature[-1]  # Get the data point with the highest feature value

            # Convert data points to indices as per gamma
            index_min = [index for index, datapoint in enumerate(X, start=1) if datapoint == i_min][0]
            index_max = [index for index, datapoint in enumerate(X, start=1) if datapoint == i_max][0]

            # Clause (9): The data point with the lowest feature value is directed left
            formula.append([-1 * get_var_index('a', t, f), get_var_index('s', index_min, t)])

            # Clause (10): The data point with the highest feature value is directed right
            formula.append([-1 * get_var_index('a', t, f), -1 * get_var_index('s', index_max, t)])

    print("Clauses from enforce_redundant_constraints:")
    for clause in formula.clauses:
        print(clause)

    return formula


def find_min_depth_tree(F, X, gamma, C, max_depth=10):
    global current_index, var_mapping, node_mapping

    for depth in range(1, max_depth + 1):
        current_index = 1  # Reset variable index for each depth
        var_mapping = {}  # Reset variable mapping for each depth
        node_mapping = defaultdict(int)
        TB = [1]  # Root node
        TL = [i for i in range(2, 2**(depth + 1))]  # Leaf nodes
        build_node_mapping(TB)

        formula = CNF()

        # Add clauses for decision tree properties
        formula.extend(choose_one_feature_at_branching(TB, F))
        formula.extend(enforce_ordering_at_branching(TB, F, X))
        formula.extend(enforce_path_validity(TL, X))
        formula.extend(enforce_leaf_node_labels(TL, C))
        formula.extend(enforce_redundant_constraints(TB, F, X))

        # Clause (11) for correct classification
        for xi_index, xi in enumerate(X, start=1):
            for t in TL:
                correct_label = gamma[xi_index]
                # Create a clause that asserts: if data point xi reaches leaf t, then t must have the correct label
                formula.append([-1 * get_var_index('z', xi_index, t), get_var_index('g', t, correct_label)])
        # Attempt to solve the CNF formula
        with Solver() as solver:
            solver.append_formula(formula)
            if solver.solve():
                # print solvable equation
                print("Formula to be solved with clauses: ")
                for clause in formula.clauses:
                    print(clause)
                # If a solution is found, print the relevant information
                print("Found solution at depth:", depth)
                # Map variable numbers back to variable names and print
                # Print all variables
                all_vars = {v: k for k, v in var_mapping.items()}
                print("Variable Mappings:")
                for var in sorted(all_vars.keys()):
                    print(f"Variable {var}: {all_vars[var]}")
                # Print only the true variables
                print("\nVariables that are True:")
                for var in sorted(all_vars.keys()):
                    if solver.get_model()[var - 1] > 0:  # PySAT models are 1-indexed
                        print(f"Variable {var} (representing {all_vars[var]}) is True")
                return solver.get_model()

        print(f"No solution at depth {depth}, trying next depth.")

    print("No solution within max depth.")
    return None

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

    min_depth_tree = find_min_depth_tree(F, X, gamma, C)
    print("Solution to problem: ")
    print(min_depth_tree)