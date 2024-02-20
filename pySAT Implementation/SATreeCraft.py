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

# from min_height_tree_module import *
# from min_height_tree_categorical_module import *
# from fixed_height_tree_module import *
# from fixed_height_tree_categorical_module import *

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
                 is_classification = True, classifciation_objective = 'min_height', fixed_depth = None, tree_structure = 'Complete',
                 ):
        
        """Initializes the SATreeCraft instance with provided dataset and configuration."""

        self.dataset = dataset
        self.features = features
        self.labels = labels
        self.true_labels_for_points = true_labels_for_points
        self.features_numerical = features_numerical
        self.features_categorical = features_categorical
        self.is_classification = is_classification
        self.classifciation_objective = classifciation_objective
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
                
                if self.classifciation_objective == 'min_height': # minimum height 100% accuracy on training problem
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
                if self.classifciation_objective == 'min_height':
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
    def export_cnf(self):
        """
        Exports the final CNF formula to a file in DIMACS format. This allows for the use
        of the CNF with external solvers. The export is only available after solving the CNF. 
        Supports both weighted and non weighted cnf. 
        """
        self.final_cnf.to_file('dimacs/export_to_solver.cnf')


 