# Created by: Haris Rasul
# Date: Feb 20th 2024
# SATreeCraft Python Library for user oriented approach
# Two Classification objectives - Min height tree 100% training classification ; Max accuracy given fixed depth 
# Works on Catgeoircal feature and Numerical feature dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from classification_problems.min_height_tree_module import *
from classification_problems.fixed_height_tree_module import *
from classification_problems.min_height_tree_categorical_module import *
from classification_problems.fixed_height_tree_categorical_module import *

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SATreeCraft:
    """
    SATreeCraft is a Python Library designed to solve classification problems
    using SAT-based decision trees. It supports datasets with categorical and/or
    numerical features and can optimize for minimum tree height or maximum accuracy
    given a fixed depth. It also supports clusteirng tree objectives such as maximizing minimum split and minimizing maximum diameter

    Attributes:
        dataset (array): The dataset to be used for tree construction.
        features (array): The list of feature names in the dataset.
        labels (array): The list of labels in the dataset.
        true_labels_for_points (array): The list of true labels for data points.
        features_numerical (array, optional): List of indices for numerical features.
        features_categorical (array, optional): List of indices for categorical features.
        is_classification (bool): Flag to indicate if the problem is classification. Default is True. Will support Clustering in future 
        classification_objective (str): The objective of the classification ('min_height' or 'max_accuracy').
        fixed_depth (int, optional): The depth of the tree if 'max_accuracy' is the classification objective.
        tree_structure (str): The type of tree structure to build ('Complete' or other types if supported in the future).

    Methods:
        solve: Determines the appropriate solving strategy based on the problem domain and objectives.
        export_cnf: Exports the final CNF formula to a DIMACS format file.
    """

    def __init__(self, dataset,features,labels, true_labels_for_points, features_numerical = None, features_categorical = None,
                 is_classification = True, classification_objective = 'min_height', fixed_depth = None, tree_structure = 'Complete',
                 ):
        
        """Initializes the SATreeCraft instance with provided dataset and configuration."""

        self.dataset = dataset
        self.features = features
        self.labels = labels
        self.true_labels_for_points = true_labels_for_points
        self.features_numerical = features_numerical
        self.features_categorical = features_categorical
        self.is_classification = is_classification
        self.classification_objective = classification_objective
        self.fixed_depth = fixed_depth
        self.tree_structure = tree_structure
        
        self.tree_model = None
        self.sat_solution = None
        self.min_cost = None
        self.min_depth = None
        self.final_cnf = None
        self.final_literals = None

    ##### Categorical Classfication Problems ####
        
    def find_min_depth_tree_categorical_problem(self, features, features_categorical, features_numerical, labels, true_labels_for_points, dataset):
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
        
        return tree_with_thresholds, literals, depth, solution, cnf
    
    def find_fixed_depth_tree_categorical_problem(self, features, features_categorical, features_numerical, labels, true_labels_for_points, dataset, depth):
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
        
        return tree_with_thresholds, literals, depth, solution, cost, wcnf

    #### Numerical Classification Problems ####
    
    def find_min_depth_tree_problem(self, features, labels, true_labels_for_points, dataset):
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
                dot.render(f'images/min_height/binary_decision_tree_min_depth_{depth}', format='png', cleanup=True)
            else:
                print('no solution at depth', depth)
                depth += 1  # Increase the depth and try again
        
        return tree_with_thresholds, literals, depth, solution, cnf
    
    def find_fixed_depth_tree_problem(self, features, labels, true_labels_for_points, dataset,depth):
        solution = "No solution exists"
        tree_with_thresholds = None
        tree = None
        literals = None
        cost = None

        tree, TB, TL = build_complete_tree(depth)
        literals = create_literals_fixed_tree(TB, TL, features, labels, len(dataset))
        wcnf = build_clauses_fixed_tree(literals, dataset, TB, TL, len(features), labels, true_labels_for_points)
        solution,cost = solve_wcnf(wcnf, literals, TL, tree, labels, features, dataset)
        
        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds(tree, literals, solution, dataset)
            dot = visualize_tree(tree_with_thresholds)
            dot.render(f'images/fixed_height/binary_decision_tree_fixed_depth_{depth}', format='png', cleanup=True)
        else:
            print('could not find solution')
            return 'No solution'
        
        return tree_with_thresholds, literals, depth, solution, cost, wcnf

    #### SAT Solving given problem ####
    def solve(self):
        """
        Solve the decision tree problem based on specified objectives and dataset features. Classifcation or Clustering 
        It chooses between categorical and numerical feature handling as well as the optimization
        objective (minimum height or maximum accuracy given a fixed depth).
        """

        if self.is_classification: # classifciation problem domain
            
            if self.features_categorical is not None and len(self.features_categorical) > 0: # categorical feature dataset
                
                if self.classification_objective == 'min_height': # minimum height 100% accuracy on training problem
                    self.model, self.final_literals, self.min_depth, self.sat_solution, self.final_cnf = self.find_min_depth_tree_categorical_problem(self.features, 
                                                                                                              self.features_categorical, 
                                                                                                              self.features_numerical, 
                                                                                                              self.labels, self.true_labels_for_points, self.dataset)
                else: # Max accuracy problem
                    self.model, self.final_literals, self.fixed_depth, self.sat_solution, self.min_cost, self.final_cnf = self.find_fixed_depth_tree_categorical_problem(self.features, 
                                                                                                              self.features_categorical, 
                                                                                                              self.features_numerical, 
                                                                                                              self.labels, 
                                                                                                              self.true_labels_for_points, 
                                                                                                              self.dataset, 
                                                                                                              self.fixed_depth)
            else: # numerical feature dataset strictly
                if self.classification_objective == 'min_height':
                    self.model, self.final_literals, self.min_depth,self.sat_solution, self.final_cnf = self.find_min_depth_tree_problem(self.features, 
                                                                                                                          self.labels, 
                                                                                                                          self.true_labels_for_points, 
                                                                                                                          self.dataset)
                else: # max accuracy problem
                    self.model, self.final_literals, self.fixed_depth, self.sat_solution, self.min_cost, self.final_cnf = self.find_fixed_depth_tree_problem(self.features, 
                                                                                                               self.labels, 
                                                                                                               self.true_labels_for_points, 
                                                                                                               self.dataset,
                                                                                                               self.fixed_depth)
    def export_cnf(self, filename='dimacs/export_to_solver.cnf'):
        """
        Exports the final CNF formula to a file in DIMACS format. This allows for the use
        of the CNF with external solvers. The export is only available after solving the CNF. 
        Supports both weighted and non weighted cnf. 
        """
        if self.final_cnf:
            self.final_cnf.to_file(filename)
        else:
            raise ValueError("Cannot export CNF. The final CNF is not available. Make sure to solve the problem first.")


    
    def export_cnf_max_accuracy_problem(self, filename='dimacs/export_to_solver_max_acc_problem.cnf'):
        """
        Exports the final CNF formula to a file in DIMACS format. This allows for the use
        of the CNF with external solvers. Export before solving max accuracy problem.
        """
        if self.is_classification: # classifciation problem domain
                if self.classification_objective != 'min_height': # minimum height 100% accuracy on training problem
                    
                    tree_with_thresholds = None
                    tree = None
                    literals = None
                    cost = None
                    tree, TB, TL = build_complete_tree(self.fixed_depth)
                    literals = create_literals_fixed_tree(TB, TL, self.features, self.labels, len(self.dataset))
                    
                    if self.features_categorical is not None and len(self.features_categorical) > 0: # categorical feature dataset
                        wcnf = build_clauses_categorical_fixed(literals, self.dataset, TB, TL, len(self.features), 
                                                            self.features_categorical, self.features_numerical, 
                                                            self.labels,self.true_labels_for_points)
                    else:
                        wcnf = build_clauses_fixed_tree(literals, self.dataset, TB, TL, len(self.features), self.labels, self.true_labels_for_points)
                    
                    wcnf.to_file(filename)
                else:
                    ("Cannot export CNF without solving for min height problem first.")
        else:
            raise ValueError("Cannot export CNF. The final CNF is not available. Make sure to solve the problem first.")
    
    def export_cnf_min_height_k(self,depth,filename = 'dimacs/export_to_solver_min_height_problem_at_given_depth.cnf'):
        """
        Exports the CNF formula at a given depth k to a file in DIMACS format. 
        This allows for External solver support of the CNF problem.
        """
        if self.is_classification: # classifciation problem domain
                if self.classification_objective == 'min_height': # minimum height 100% accuracy on training problem
                    solution = "No solution exists"
                    tree_with_thresholds = None
                    tree = None
                    literals = None
                    tree, TB, TL = build_complete_tree(depth)
                    literals = create_literals(TB, TL, self.features, self.labels, len(self.dataset))

                    if self.features_categorical is not None and len(self.features_categorical) > 0: # categorical feature dataset
                        cnf = build_clauses_categorical(literals, 
                                                        self.dataset, TB, TL, len(self.features), 
                                                        self.features_categorical, self.features_numerical, self.labels, self.true_labels_for_points)
                    else:
                        cnf = build_clauses(literals, self.dataset, TB, TL, len(self.features), self.labels, self.true_labels_for_points)
                    cnf.to_file(filename)
                else:
                    ("Must be a min height objective")
        else:
            ("Cannot export CNF ")
    
    
    def create_solution_matrix(self, literals, solution, var_type):
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
    
    
    def display_solution(self):
        '''
        Display solved solution of porblem in readble format of literals
        '''
        print("\nSolution of Literals")
        
        if self.classification_objective == 'min_height':
            var_types = ['a', 's', 'z', 'g']
        else:
            var_types = ['a', 's', 'z', 'g','p']
        for var_type in var_types:
            if var_type != 'p':
                matrix = create_solution_matrix(self.final_literals, self.sat_solution, var_type)
                print(f"{var_type.upper()} Variables:")
                for row in matrix:
                    print(' '.join(map(str, row)))
                print("\n")
            elif var_type == 'p' and self.classification_objective != 'min_height':
                # finish the p_literals 
                print("P Variables:")
                for p_literal, value in self.final_literals.items():
                    if p_literal.startswith('p_'):
                        # Convert positive values to 1 and negative to 0
                        clue = 1 if value > 0 else 0
                        print(f"{p_literal}: {clue}")
