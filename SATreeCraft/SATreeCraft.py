# Created by: Haris Rasul
# Date: Feb 20th 2024
# SATreeCraft Python Library for user oriented approach
# Two Classification objectives - Min height tree 100% training classification ; Max accuracy given fixed depth 
# Works on Catgeoircal feature and Numerical feature dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import math
# clasification modules 
from classification_problems.min_height_tree_module import *
from classification_problems.fixed_height_tree_module import *
from classification_problems.min_height_tree_categorical_module import *
from classification_problems.fixed_height_tree_categorical_module import *

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from classification_problems.additional_classification_constraints import *

# clustering modules 
from clustering_problems.clustering_advanced import *
from clustering_problems.clustering_minsplit import *
from clustering_problems.clustering_smartPairs import *

# Loandra solver support 
from loandra_support.loandra import *

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
        tree_structure (str): The type of tree structure to build ('Complete' or 'Oblivious').

    Methods:
        solve: Determines the appropriate solving strategy based on the problem domain and objectives.
        export_cnf: Exports the final CNF formula to a DIMACS format file.
    """

    def __init__(self, dataset,features,labels = None, true_labels_for_points = None, features_numerical = None, features_categorical = None,
                 is_classification = True, classification_objective = 'min_height', fixed_depth = None, tree_structure = 'Complete', min_support = 0,
                 min_margin = 1, k_clusters = None, clustering_objective = 'max_diameter', is_clustering = False, epsilon = 0, CL_pairs = np.array([]), ML_pairs = np.array([]),
                 smart_pairs = False
                 ):
        
        """Initializes the SATreeCraft instance with provided dataset and configuration."""

        self.dataset = dataset
        self.features = features
        
        # Classification tools 
        self.labels = labels
        self.true_labels_for_points = true_labels_for_points
        self.features_numerical = features_numerical
        self.features_categorical = features_categorical
        self.is_classification = is_classification
        self.classification_objective = classification_objective
        self.fixed_depth = fixed_depth
        self.tree_structure = tree_structure
        self.min_support = min_support
        self.min_margin = min_margin


        # Clustering tools
        self.k_clusters = k_clusters
        self.clustering_objective = clustering_objective
        self.is_clustering = is_clustering
        
        if self.is_clustering or (self.k_clusters is not None):
            self.is_classification = False
        
        self.epsilon = epsilon
        self.CL_pairs = CL_pairs
        self.ML_pairs = ML_pairs
        self.smart_pairs = smart_pairs

        # return types 
        self.tree_model = None
        self.sat_solution = None
        self.min_cost = None
        self.min_depth = None
        self.final_cnf = None
        self.final_literals = None
        self.cluster_assignments = None
        self.cluster_diameters = None

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

            # Oblivious Tree Constraints addition if ever used 
            if (self.tree_structure == 'Oblivious'):
                cnf = add_oblivious_tree_constraints(cnf,features,depth,literals)
            if(self.tree_structure == 'Oblivious2'):
                cnf = add_oblivious_tree_constraints2(cnf,features,depth,literals,dataset)

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

        # Min support constraint can be added
        if (self.min_support > 0):
            wcnf = min_support(wcnf, literals, dataset, TL, self.min_support)
        
        # Oblivious Tree Structure Encodings enforced assuming not complete 
        if (self.tree_structure == 'Oblivious'):
            wcnf = add_oblivious_tree_constraints(wcnf,features,depth,literals)
        if(self.tree_structure == 'Oblivious2'):
            wcnf = add_oblivious_tree_constraints2(wcnf,features,depth,literals,dataset)


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
            
            # Oblivious Tree Constraints addition if ever used 
            if (self.tree_structure == 'Oblivious'):
                cnf = add_oblivious_tree_constraints(cnf,features,depth,literals)
            if(self.tree_structure == 'Oblivious2'):
                cnf = add_oblivious_tree_constraints2(cnf,features,depth,literals,dataset)


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

        # min margin constraint - only for numerical problem 
        if (self.min_margin > 1):
            wcnf = build_clauses_fixed_tree_min_margin_constraint_add(literals, dataset, TB, TL, len(features), labels, true_labels_for_points, self.min_margin)
    
        else:
            wcnf = build_clauses_fixed_tree(literals, dataset, TB, TL, len(features), labels, true_labels_for_points)

        # min support constraint 
        if (self.min_support > 0):
            wcnf = min_support(wcnf, literals, dataset, TL, self.min_support)
        
        # Oblivious Tree Structure Encodings enforced assuming not complete 
        if (self.tree_structure == 'Oblivious'):
            # print("adding oblivious contraints")
            wcnf = add_oblivious_tree_constraints(wcnf,features,depth,literals)
        if(self.tree_structure == 'Oblivious2'):
            wcnf = add_oblivious_tree_constraints2(wcnf,features,depth,literals,dataset)

        solution,cost = solve_wcnf(wcnf, literals, TL, tree, labels, features, dataset)
        
        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds(tree, literals, solution, dataset)
            dot = visualize_tree(tree_with_thresholds)
            dot.render(f'images/fixed_height/binary_decision_tree_fixed_depth_{depth}', format='png', cleanup=True)
        else:
            print('could not find solution')
            return 'No solution'
        
        return tree_with_thresholds, literals, depth, solution, cost, wcnf

    
    # Save the plot to the specified directory with the given filename format
    def plot_and_save_clusters_to_drive(self,dataset, cluster_assignments, k_clusters):
        """
        Plots the dataset points before and after clustering if the dataset has 1 or 2 features.
        Creates a side-by-side plot showing the dataset before clustering and after with cluster IDs.
        Saves the plot to the specified directory with a filename based on the number of clusters.
        Does not display the plot in the output.

        Parameters:
        - dataset (np.ndarray): The original dataset with data points.
        - cluster_assignments (dict): A dictionary with cluster IDs and lists of data points in each cluster.
        - k_clusters (int): The number of clusters.

        Returns:
        - full_path (str): The path to the saved plot image.
        """
        # Define the directory and filename
        directory = 'images/cluster_trees/'
        filename = f'cluster_tree_with_cluster_size{k_clusters}.png'
        full_path = directory + filename

        # Create the plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.patch.set_facecolor('white')
        
        if dataset.shape[1] == 2:  # If 2D dataset
            axes[0].scatter(dataset[:, 0], dataset[:, 1], c='gray', label='Data Points')
            axes[0].set_title('Before Clustering')
            axes[1].scatter(dataset[:, 0], dataset[:, 1], c='gray', label='Data Points')
            axes[1].set_title('After Clustering')
        elif dataset.shape[1] == 1:  # If 1D dataset
            axes[0].scatter(dataset[:, 0], np.zeros_like(dataset[:, 0]), c='gray', label='Data Points')
            axes[0].set_title('Before Clustering')
            axes[1].scatter(dataset[:, 0], np.zeros_like(dataset[:, 0]), c='gray', label='Data Points')
            axes[1].set_title('After Clustering')
        else:
            return 'can only plot 2d or 1d datasets'

        # Assign colors to clusters
        colors = plt.cm.tab10(np.linspace(0, 1, k_clusters))
        for cluster_id, data_points in cluster_assignments.items():
            if dataset.shape[1] == 2:
                axes[1].scatter(dataset[data_points, 0], dataset[data_points, 1], 
                                color=colors[cluster_id], label=f'Cluster {cluster_id}')
            elif dataset.shape[1] == 1:
                axes[1].scatter(dataset[data_points, 0], np.zeros_like(dataset[data_points, 0]), 
                                color=colors[cluster_id], label=f'Cluster {cluster_id}')

        # Add legend to the second plot
        axes[1].legend()

        # Save the figure
        fig.savefig(full_path)
        plt.close(fig)  # Close the figure to prevent it from displaying in the output
        return full_path

    def solve_clustering_problem_max_diameter(self, dataset,features,k_clusters, depth, epsilon, CL_pairs, ML_pairs):
        dataset_size = len(dataset)
        num_features = len(features)
        dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
        tree_structure, TB, TL = build_complete_tree_clustering(depth)
        
        literals = create_literals_cluster_tree(TB, TL, features, k_clusters, dataset_size,distance_classes)
        
        wcnf = build_clauses_cluster_tree_MD(literals, dataset, TB, TL, num_features, k_clusters,
                                    CL_pairs, ML_pairs, distance_classes)
    
        solution = solve_wcnf_clustering(wcnf)
        a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector = create_literal_matrices(literals=literals,
                                                                                                    solution=solution,
                                                                                                    dataset_size=len(dataset),
                                                                                                    k_clusters=k_clusters,
                                                                                                    TB=TB,
                                                                                                    TL=TL,
                                                                                                    num_features=len(features),
                                                                                                    distance_classes= distance_classes
                                                                                                    )
        cluster_assignments, cluster_diameters = assign_clusters_and_diameters(x_i_c_matrix, dataset, k_clusters)
        if (len(self.features) <= 2):
            self.plot_and_save_clusters_to_drive(dataset, cluster_assignments, k_clusters)
        return cluster_assignments, cluster_diameters, literals, solution


    def solve_clustering_problem_bicriteria(self, dataset,features, k_clusters, depth, epsilon, CL_pairs, ML_pairs):
        dataset_size = len(dataset)
        num_features = len(features)
        dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
        tree_structure, TB, TL = build_complete_tree_clustering(depth)
        
        literals = create_literals_cluster_tree_bicriteria(TB, TL, features, k_clusters, dataset_size,distance_classes)
       
        if self.smart_pairs:
            wcnf = build_clauses_cluster_tree_MD_MS_Smart_Pair(literals, dataset, TB, TL, num_features, k_clusters,
                                    CL_pairs, ML_pairs, distance_classes)
        else:
            wcnf = build_clauses_cluster_tree_MD_MS(literals, dataset, TB, TL, num_features, k_clusters,
                                    CL_pairs, ML_pairs, distance_classes)
    
        solution = solve_wcnf_clustering(wcnf)
        a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector, bw_p_vector = create_literal_matrices_bicriteria(
                                                                                                                            literals=literals,
                                                                                                                            solution=solution,
                                                                                                                            dataset_size=len(dataset),
                                                                                                                            k_clusters=k_clusters,
                                                                                                                            TB=TB,
                                                                                                                            TL=TL,
                                                                                                                            num_features=len(features),
                                                                                                                            distance_classes= distance_classes
                                                                                                                            )
        cluster_assignments, cluster_diameters = assign_clusters_and_diameters(x_i_c_matrix, dataset, k_clusters)
        if (len(self.features) <= 2):
            self.plot_and_save_clusters_to_drive(dataset, cluster_assignments, k_clusters)
        # print(x_i_c_matrix)
        # print(literals)
        return cluster_assignments, cluster_diameters, literals, solution         
    
    #### SAT Solving given problem ####
    def solve(self):
        """
        Solve the decision tree problem based on specified objectives and dataset features. Classifcation or Clustering 
        It chooses between categorical and numerical feature handling as well as the optimization
        objective (minimum height or maximum accuracy given a fixed depth).
        must set is_classification to false to work on clusteirng porblem or set is_clustering to true 
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
        else:
            max_clusters = 2 ** self.fixed_depth
            if self.k_clusters > max_clusters:
                raise ValueError(f"The assigned depth {self.fixed_depth} is not sufficient to accommodate {self.k_clusters} clusters.")
            
            if self.clustering_objective == 'max_diameter':
                self.cluster_assignments, self.cluster_diameters, self.final_literals, self.sat_solution = self.solve_clustering_problem_max_diameter(self.dataset, 
                                                                                                                                                    self.features, 
                                                                                                                                                    self.k_clusters, 
                                                                                                                                                    self.fixed_depth, 
                                                                                                                                                    self.epsilon, 
                                                                                                                                                    self.CL_pairs, 
                                                                                                                                                    self.ML_pairs)
            else: # Bicriteria
                # print('solving bicriteria') 
                self.cluster_assignments, self.cluster_diameters, self.final_literals, self.sat_solution = self.solve_clustering_problem_bicriteria(self.dataset, 
                                                                                                                                                    self.features, 
                                                                                                                                                    self.k_clusters, 
                                                                                                                                                    self.fixed_depth, 
                                                                                                                                                    self.epsilon, 
                                                                                                                                                    self.CL_pairs, 
                                                                                                                                                    self.ML_pairs)

    ############################## LOANDRA Functionality Support for External SOLVING ###################################
    
    
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

    def export_cnf_min_height(self, filename='dimacs/export_to_solver_min_height.cnf'):
        """
        Exports the final CNF formula to a file in DIMACS format. This allows for the use
        of the CNF with external solvers. The export is only available after solving the CNF. 
        Supports both weighted and non weighted cnf. 
        """
        if self.final_cnf:
            wcnf = WCNF()
            for clause in self.final_cnf:
                    wcnf.append(clause)
            wcnf.to_file(filename)
        else:
            raise ValueError("Cannot export CNF. The final CNF is not available")
    
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
                    wcnf = WCNF()
                    for clause in cnf:
                        wcnf.append(clause)
                    wcnf.to_file(filename)
                else:
                    ("Must be a min height objective")
        else:
            ("Cannot export CNF ")
    
    
    def find_fixed_depth_tree_problem_loandra(self, features, labels, true_labels_for_points, dataset, depth, loandra_path, execution_path):
        solution = "No solution exists"
        tree_with_thresholds = None
        tree = None
        literals = None
        cost = None

        tree, TB, TL = build_complete_tree(depth)
        literals = create_literals_fixed_tree(TB, TL, features, labels, len(dataset))

        # min margin constraint - only for numerical problem 
        if (self.min_margin > 1):
            wcnf = build_clauses_fixed_tree_min_margin_constraint_add(literals, dataset, TB, TL, len(features), labels, true_labels_for_points, self.min_margin)
    
        else:
            wcnf = build_clauses_fixed_tree(literals, dataset, TB, TL, len(features), labels, true_labels_for_points)

        # min support constraint 
        if (self.min_support > 0):
            wcnf = min_support(wcnf, literals, dataset, TL, self.min_support)
        
        # Oblivious Tree Structure Encodings enforced assuming not complete 
        if (self.tree_structure == 'Oblivious'):
            # print("adding oblivious contraints")
            wcnf = add_oblivious_tree_constraints(wcnf,features,depth,literals)
        if (self.tree_structure == 'Oblivious2'):
            # print("adding oblivious contraints")
            wcnf = add_oblivious_tree_constraints2(wcnf,features,depth,literals,dataset)

        
        # LOANDRA SUPPORT - EXPORT FILE 
        wcnf.to_file(execution_path)
        solution,cost = run_loandra_and_parse_results(loandra_path, execution_path)
        solution = transform_tree_from_loandra(solution, literals, TL, tree, labels,features,dataset)


        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds(tree, literals, solution, dataset)
            dot = visualize_tree(tree_with_thresholds)
            dot.render(f'images/fixed_height/LOANDRA_SOLVED_binary_decision_tree_fixed_depth_{depth}', format='png', cleanup=True)
        else:
            print('could not find solution')
            return 'No solution'
        
        return tree_with_thresholds, literals, depth, solution, cost, wcnf


    def find_fixed_depth_tree_categorical_problem_loandra(self, features, features_categorical, features_numerical, labels, true_labels_for_points, 
                                                          dataset, depth, loandra_path, execution_path):
        solution = "No solution exists"
        tree_with_thresholds = None
        tree = None
        literals = None
        cost = None

        tree, TB, TL = build_complete_tree(depth)
        literals = create_literals_fixed_tree(TB, TL, features, labels, len(dataset))

        wcnf = build_clauses_categorical_fixed(literals, dataset, TB, TL, len(features), features_categorical, features_numerical, labels,true_labels_for_points)

        # Min support constraint can be added
        if (self.min_support > 0):
            wcnf = min_support(wcnf, literals, dataset, TL, self.min_support)
        
        # Oblivious Tree Structure Encodings enforced assuming not complete 
        if (self.tree_structure == 'Oblivious'):
            wcnf = add_oblivious_tree_constraints(wcnf,features,depth,literals)
        if (self.tree_structure == 'Oblivious2'):
            wcnf = add_oblivious_tree_constraints2(wcnf,features,depth,literals,dataset)

        # LOANDRA SUPPORT - EXPORT FILE 
        wcnf.to_file(execution_path)
        solution,cost = run_loandra_and_parse_results(loandra_path, execution_path)
        solution = transform_tree_from_loandra(solution, literals, TL, tree, labels,features,dataset)
        
        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds_categorical(tree, literals, solution, dataset, features_categorical)
            dot = visualize_tree(tree_with_thresholds)
            dot.render(f'images/fixed_height/LOANDRA_SOLVED_binary_decision_tree_fixed_with_categorical_features_depth_{depth}', format='png', cleanup=True)
        else:
            print('could not find solution')
            return 'No solution'
        
        return tree_with_thresholds, literals, depth, solution, cost, wcnf
    

    def find_min_depth_tree_problem_loandra(self, features, labels, true_labels_for_points, dataset,loandra_path,execution_path):
        depth = 1  # Start with a depth of 1
        solution = "No solution exists"
        tree_with_thresholds = None
        tree = None
        literals = None

        while solution == "No solution exists":
            tree, TB, TL = build_complete_tree(depth)
            literals = create_literals(TB, TL, features, labels, len(dataset))
            cnf = build_clauses(literals, dataset, TB, TL, len(features), labels, true_labels_for_points)
            
            # Oblivious Tree Constraints addition if ever used 
            if (self.tree_structure == 'Oblivious'):
                cnf = add_oblivious_tree_constraints(cnf,features,depth,literals)
            if (self.tree_structure == 'Oblivious2'):
                cnf = add_oblivious_tree_constraints2(cnf,features,depth,literals,dataset)
            
            # LOANDRA SUPPORT - EXPORT FILE
            
            # LOANDRA SUPPORT - NEED TO CONNVERT TO NCF MAX SAT PROBLEM BASED ON THEIR IMPLMENTATION (ALL HARD CLAUSES)
            wcnf = WCNF()
            for clause in cnf:
                wcnf.append(clause)             
            wcnf.to_file(execution_path)
            
            
            solution,cost = run_loandra_and_parse_results(loandra_path, execution_path)
            
            # NO SOLUTION FOUND FROM MAX SAT - BECAUSE SCORE IS NOT ZERO!
            if cost != 0:
                solution = "No solution exists"
            
            if solution != "No solution exists":
                solution = transform_tree_from_loandra(solution, literals, TL, tree, labels,features,dataset)
                tree_with_thresholds = add_thresholds(tree, literals, solution, dataset)
                dot = visualize_tree(tree_with_thresholds)
                dot.render(f'images/min_height/LOANDRA_SOLVED_binary_decision_tree_min_depth_{depth}', format='png', cleanup=True)
            else:
                print('no solution at depth', depth)
                depth += 1  # Increase the depth and try again
        
        return tree_with_thresholds, literals, depth, solution, cnf


    def find_min_depth_tree_categorical_problem_loandra(self, features, features_categorical, features_numerical, labels, 
                                                        true_labels_for_points, dataset, loandra_path, execution_path):
        depth = 1  # Start with a depth of 1
        solution = "No solution exists"
        tree_with_thresholds = None
        tree = None
        literals = None

        while solution == "No solution exists":
            tree, TB, TL = build_complete_tree(depth)
            literals = create_literals(TB, TL, features, labels, len(dataset))
            cnf = build_clauses_categorical(literals, dataset, TB, TL, len(features), features_categorical, features_numerical, labels, true_labels_for_points)

            # Oblivious Tree Constraints addition if ever used 
            if (self.tree_structure == 'Oblivious'):
                cnf = add_oblivious_tree_constraints(cnf,features,depth,literals)
            if (self.tree_structure == 'Oblivious2'):
                cnf = add_oblivious_tree_constraints2(cnf,features,depth,literals,dataset)

            # LOANDRA SUPPORT - NEED TO CONNVERT TO NCF MAX SAT PROBLEM BASED ON THEIR IMPLMENTATION (ALL HARD CLAUSES)
            wcnf = WCNF()
            for clause in cnf:
                wcnf.append(clause)             
            wcnf.to_file(execution_path)
            solution,cost = run_loandra_and_parse_results(loandra_path, execution_path)

            # NO SOLUTION FOUND FROM MAX SAT - BECAUSE SCORE IS NOT ZERO!
            if cost != 0:
                solution = "No solution exists"
            
            if solution != "No solution exists":
                solution = transform_tree_from_loandra(solution, literals, TL, tree, labels,features,dataset)
                tree_with_thresholds = add_thresholds_categorical(tree, literals, solution, dataset, features_categorical)
                dot = visualize_tree(tree_with_thresholds)
                dot.render(f'images/min_height/LOANDRA_SOLVED_binary_decision_tree_min_depth_with_categorical_features_depth_{depth}', format='png', cleanup=True)
            else:
                print("No solution at depth: ", depth)
                depth += 1  # Increase the depth and try again
        
        return tree_with_thresholds, literals, depth, solution, cnf


    def solve_clustering_problem_max_diameter_loandra(self, dataset,features,k_clusters, depth, epsilon, CL_pairs, ML_pairs,
                                                      loandra_path,execution_path):
        dataset_size = len(dataset)
        num_features = len(features)
        dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
        tree_structure, TB, TL = build_complete_tree_clustering(depth)
        
        literals = create_literals_cluster_tree(TB, TL, features, k_clusters, dataset_size,distance_classes)
        wcnf = build_clauses_cluster_tree_MD(literals, dataset, TB, TL, num_features, k_clusters,
                                    CL_pairs, ML_pairs, distance_classes)
    
        wcnf.to_file(execution_path)
       
        solution,cost = run_loandra_and_parse_results(loandra_path, execution_path)
        
        a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector = create_literal_matrices(literals=literals,
                                                                                                    solution=solution,
                                                                                                    dataset_size=len(dataset),
                                                                                                    k_clusters=k_clusters,
                                                                                                    TB=TB,
                                                                                                    TL=TL,
                                                                                                    num_features=len(features),
                                                                                                    distance_classes= distance_classes
                                                                                                    )
        cluster_assignments, cluster_diameters = assign_clusters_and_diameters(x_i_c_matrix, dataset, k_clusters)
        if (len(self.features) <= 2):
            self.plot_and_save_clusters_to_drive(dataset, cluster_assignments, k_clusters)
        
        return cluster_assignments, cluster_diameters, literals, solution

    def solve_clustering_problem_bicriteria_loandra(self, dataset,features,k_clusters, depth, epsilon, CL_pairs, ML_pairs,
                                                      loandra_path,execution_path):
        dataset_size = len(dataset)
        num_features = len(features)
        dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
        tree_structure, TB, TL = build_complete_tree_clustering(depth)
        
        literals = create_literals_cluster_tree_bicriteria(TB, TL, features, k_clusters, dataset_size,distance_classes)
        
        if self.smart_pairs:
            wcnf = build_clauses_cluster_tree_MD_MS_Smart_Pair(literals, dataset, TB, TL, num_features, k_clusters,
                                    CL_pairs, ML_pairs, distance_classes)
        else:
            wcnf = build_clauses_cluster_tree_MD_MS(literals, dataset, TB, TL, num_features, k_clusters,
                                    CL_pairs, ML_pairs, distance_classes)
    
        wcnf.to_file(execution_path)
       
        solution,cost = run_loandra_and_parse_results(loandra_path, execution_path)
        
        a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector, bw_p_vector = create_literal_matrices_bicriteria(
                                                                                                                            literals=literals,
                                                                                                                            solution=solution,
                                                                                                                            dataset_size=len(dataset),
                                                                                                                            k_clusters=k_clusters,
                                                                                                                            TB=TB,
                                                                                                                            TL=TL,
                                                                                                                            num_features=len(features),
                                                                                                                            distance_classes= distance_classes
                                                                                                                            )
        cluster_assignments, cluster_diameters = assign_clusters_and_diameters(x_i_c_matrix, dataset, k_clusters)
        if (len(self.features) <= 2):
            self.plot_and_save_clusters_to_drive(dataset, cluster_assignments, k_clusters)
        return cluster_assignments, cluster_diameters, literals, solution  


    def solve_loandra(self,loandra_path,execution_path='dimacs/export_to_solver.cnf'):
        """
        Solve the decision tree problem based on specified objectives and dataset features. Classifcation or Clustering 
        It chooses between categorical and numerical feature handling as well as the optimization
        objective (minimum height or maximum accuracy given a fixed depth).
        LOANDRA VARIANT - calls external solver support system 
        """

        if self.is_classification: # classifciation problem domain
            
            if self.features_categorical is not None and len(self.features_categorical) > 0: # categorical feature dataset
                
                if self.classification_objective == 'min_height': # minimum height 100% accuracy on training problem
                    self.model, self.final_literals, self.min_depth, self.sat_solution, self.final_cnf = self.find_min_depth_tree_categorical_problem_loandra(self.features, 
                                                                                                              self.features_categorical, 
                                                                                                              self.features_numerical, 
                                                                                                              self.labels, self.true_labels_for_points, self.dataset,
                                                                                                              loandra_path,
                                                                                                              execution_path)
                else: # Max accuracy problem
                    self.model, self.final_literals, self.fixed_depth, self.sat_solution, self.min_cost, self.final_cnf = self.find_fixed_depth_tree_categorical_problem_loandra(self.features, 
                                                                                                              self.features_categorical, 
                                                                                                              self.features_numerical, 
                                                                                                              self.labels, 
                                                                                                              self.true_labels_for_points, 
                                                                                                              self.dataset, 
                                                                                                              self.fixed_depth,
                                                                                                              loandra_path,
                                                                                                              execution_path)
            else: # numerical feature dataset strictly
                if self.classification_objective == 'min_height':
                    self.model, self.final_literals, self.min_depth,self.sat_solution, self.final_cnf = self.find_min_depth_tree_problem_loandra(self.features, 
                                                                                                                          self.labels, 
                                                                                                                          self.true_labels_for_points, 
                                                                                                                          self.dataset,
                                                                                                                          loandra_path,
                                                                                                                          execution_path)
                else: # max accuracy problem
                    self.model, self.final_literals, self.fixed_depth, self.sat_solution, self.min_cost, self.final_cnf = self.find_fixed_depth_tree_problem_loandra(self.features, 
                                                                                                               self.labels, 
                                                                                                               self.true_labels_for_points, 
                                                                                                               self.dataset,
                                                                                                               self.fixed_depth,
                                                                                                               loandra_path,
                                                                                                               execution_path)
        else:
            max_clusters = 2 ** self.fixed_depth
            if self.k_clusters > max_clusters:
                raise ValueError(f"The assigned depth {self.fixed_depth} is not sufficient to accommodate {self.k_clusters} clusters.")
            
            if self.clustering_objective == 'max_diameter':
                self.cluster_assignments, self.cluster_diameters, self.final_literals, self.sat_solution = self.solve_clustering_problem_max_diameter_loandra(self.dataset, 
                                                                                                                                                    self.features, 
                                                                                                                                                    self.k_clusters, 
                                                                                                                                                    self.fixed_depth, 
                                                                                                                                                    self.epsilon, 
                                                                                                                                                    self.CL_pairs, 
                                                                                                                                                    self.ML_pairs,
                                                                                                                                                    loandra_path,
                                                                                                                                                    execution_path
                                                                                                                                                    )
            else: # bicriteria
                # print('solving bicriteria')
                self.cluster_assignments, self.cluster_diameters, self.final_literals, self.sat_solution = self.solve_clustering_problem_bicriteria_loandra(self.dataset, 
                                                                                                                                                    self.features, 
                                                                                                                                                    self.k_clusters, 
                                                                                                                                                    self.fixed_depth, 
                                                                                                                                                    self.epsilon, 
                                                                                                                                                    self.CL_pairs, 
                                                                                                                                                    self.ML_pairs,
                                                                                                                                                    loandra_path,
                                                                                                                                                    execution_path
                                                                                                                                                    )

 
    ##################################### Auxillary Helper Functions for User Interface #############################


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
