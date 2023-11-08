### Made by Haris Rasul
### Oct 15th 2023
### Purpose: SAT Encodings
# sat_encoding.py
from pysat.formula import CNF
from pysat.solvers import Solver
from collections import defaultdict
from DecisionTree import * 
node_mapping = defaultdict(int)

# Clauses for minimal height optimal Decision:


# A dictionary to store the mapping between (prefix, t, j) tuples and unique variable indices
var_mapping = {}
current_index = 1

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



def calculate_median_threshold(feature_values):
    sorted_values = sorted(feature_values)
    mid_point = len(sorted_values) // 2
    # If even number of elements, take the average of the middle two
    if len(sorted_values) % 2 == 0:
        return (sorted_values[mid_point - 1] + sorted_values[mid_point]) / 2.0
    # If odd, take the middle element
    else:
        return sorted_values[mid_point]


def decode_solution_to_tree(model, TB, TL, F, X, gamma):
    # Initialize nodes and data points associated with each node
    nodes = {t: create_tree_node(t, is_leaf=(t in TL)) for t in TB + TL}
    node_data_points = {t: X[:] for t in TB}  # Initially, all data points reach each branching node

    # Decode feature selection for branching nodes
    for t in TB:
        for f in F:
            var = get_var_index('a', t, f)
            if model[var - 1] > 0:  # Check if the variable is True in the model
                nodes[t].feature = f
                # Calculate the splitting threshold based on the data points that reach node `t`
                feature_values = [x[f] for x in node_data_points[t]]
                nodes[t].threshold = calculate_median_threshold(feature_values)
                # Split the data points for the children nodes based on the threshold
                left_data_points = [x for x in node_data_points[t] if x[f] <= nodes[t].threshold]
                right_data_points = [x for x in node_data_points[t] if x[f] > nodes[t].threshold]
                node_data_points[2*t] = left_data_points
                node_data_points[2*t + 1] = right_data_points
                break  # Only one feature should be selected per branching node

    # Decode leaf node labels
    for t in TL:
        for label in gamma.values():
            var = get_var_index('g', t, label)
            if model[var - 1] > 0:  # Check if the variable is True in the model
                nodes[t].label = label
                break  # Only one label should be assigned per leaf node

    # Link nodes to form the tree
    for t in TB + TL:
        if t in TB:  # If it's a branching node, set its children
            nodes[t].set_children(nodes.get(2*t), nodes.get(2*t + 1))
        if t != 1:  # If it's not the root, set its parent
            nodes[t].set_parent(nodes.get(t//2))

    # Assuming node 1 is the root
    return nodes[1]



if __name__ == "__main__":
    ## I am making a tree structure here randomly 
    TB = [1, 2]  # Branching nodes
    TL = [3, 4, 5]  # Leaf nodes

    # features
    F = ['F1', 'F2']

    # Toy dataset haris made 
     # Toy dataset haris made 
    # New toy dataset
    X = [
    {'F1': 5, 'F2': 3},  # Data point 1
    {'F1': 7, 'F2': 4},  # Data point 2
    {'F1': 2, 'F2': 6},  # Data point 3
    {'F1': 3, 'F2': 5},  # Data point 4
    ]

    # Labels for the new dataset
    gamma = {
        1: 1,  # Label for data point 1
        2: 1,  # Label for data point 2
        3: 2,  # Label for data point 3
        4: 3,  # Label for data point 4
    }

    # class labels
    C = [1, 2, 3]

    # Build the node mapping for the tree
    build_node_mapping(TB)

    # Now node_mapping should be populated, let's print it to check
    print("Node Mapping:")
    for key, value in node_mapping.items():
        print(f"Node {key} is a child of Node {value}")



    # CNF clauses buildout 
    cnf = create_cnf_for_decision_tree(TB, TL, F, X, C, gamma)

    
    print("Variable Mappings:")
    for key, value in var_mapping.items():
        print(f"{value}: {key}")
    
    
    # Use the SAT solver to find a solution
    solver = Solver()
    solver.append_formula(cnf)
    solution_exists = solver.solve()

    # results 
    if solution_exists:
        model = solver.get_model()
        print("SAT solver found a solution:")
        print(model)
    else:
        print("No solution exists for the given problem.")
    
    print("SAT Solver Model:")
    for variable in model:
        if variable > 0:  # Only print variables that are True in the model
            key = next(key for key, value in var_mapping.items() if value == variable)
            print(f"{variable}: {key}")


    for xi_index, xi in enumerate(X, start=1):  # start=1 to match the gamma indices
        for t in TL:
            z_var = get_var_index('z', xi_index, t)
            g_var = get_var_index('g', t, gamma[xi_index])
            if model[z_var - 1] > 0:  # If data point reaches the leaf
                print(f"Data point {xi_index} reaches leaf {t}")
                if model[g_var - 1] > 0:  # If leaf has the correct label
                    print(f"Leaf {t} correctly labeled with {gamma[xi_index]}")
                else:
                    print(f"Leaf {t} incorrectly labeled or not labeled")



    root = decode_solution_to_tree(model, TB, TL, F, X, gamma)
    #print(root)
    tree = DecisionTree(root)

    # Test predictions

    all_correct = True
    for i, x in enumerate(X):
        prediction = tree.predict(x)
        actual_label = gamma[i+1]
        is_correct = (prediction == actual_label)
        all_correct &= is_correct  #Uni testsinfs
        print(f"Data point {i+1}: {x}, Predicted label: {prediction}, Actual label: {actual_label}, Correct: {is_correct}")

    
    accuracy = tree.calculate_accuracy(X, gamma)
    print(f"Accuracy of the decision tree: {accuracy * 100:.2f}%")

    # Haris Testing 
    #assert all_correct, "Not all predictions are correct!"
    #assert accuracy == 1.0, "The accuracy is not 100%!"

    tree.visualize()