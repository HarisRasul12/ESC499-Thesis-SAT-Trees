# Created by: Haris Rasul
# Date: Feb 20th 2024
# SAT Tree model classifier 

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from classification_problems.min_height_tree_module import *
from classification_problems.fixed_height_tree_module import *
from classification_problems.min_height_tree_categorical_module import *
from classification_problems.fixed_height_tree_categorical_module import *


class SATreeClassifier:
    
    def __init__(self, tree):
        """
        A classifier that uses a pre-built decision tree to make predictions
        and evaluate performance. The tree is based off the SAT solution for training dataset

        Parameters:
        - tree (dict): Pre-built decision tree structure from SAT solution.
        """
        self.tree_model = tree

    def predict(self, data):
        """
        Predicts the labels for the given data.

        Args:
        - data (array-like): The input data for which to predict labels. Each row corresponds to a single data point.

        Returns:
        - predictions (numpy array): The predicted labels for the input data.
        """
        predictions = []

        # If data is a single sample, reshape it to be two-dimensional
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Iterate over each data point
        for point in data:
            node_index = 0  # start from the root of the tree, which is at index 0 of the tree_model list
            while self.tree_model[node_index]['type'] != 'leaf':
                # Use the feature index as an integer to access the feature value
                feature_index = int(self.tree_model[node_index]['feature'])
                feature_value = point[feature_index]

                # Determine the next node based on the feature value
                if isinstance(self.tree_model[node_index]['threshold'], list):  # categorical node
                    if feature_value in self.tree_model[node_index]['threshold']:
                        # Move to the left child
                        node_index = self.tree_model[node_index]['children'][0]
                    else:
                        # Move to the right child
                        node_index = self.tree_model[node_index]['children'][1]
                else:  # numerical node
                    # print(feature_value, self.tree_model[node_index]['threshold'])
                    if float(feature_value) <= self.tree_model[node_index]['threshold']:
                        # Move to the left child
                        node_index = self.tree_model[node_index]['children'][0]
                    else:
                        # Move to the right child
                        node_index = self.tree_model[node_index]['children'][1]

            # Once a leaf node is reached, use its label for the prediction
            predictions.append(self.tree_model[node_index]['label'])
        
        # Return predictions as a numpy array
        return np.array(predictions)
    
    def score(self, X, y_true):
        """
        Calculates the accuracy of the model.

        Parameters:
        - X (array-like): The input features for which to predict labels.
        - y_true (array-like): The true labels.

        Returns:
        - score (float): Accuracy of the model on the given data.
        """
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)
    
    def get_classification_report(self, X, y_true):
        """
        Generates a classification report.

        Parameters:
        - X (array-like): The input features for which to predict labels.
        - y_true (array-like): The true labels.

        Returns:
        - report (str): Text summary of the precision, recall, F1 score for each class.
        """
        y_pred = self.predict(X)
        return classification_report(y_true, y_pred)

    def get_confusion_matrix(self, X, y_true):
        """
        Computes the confusion matrix.

        Parameters:
        - X (array-like): The input features for which to predict labels.
        - y_true (array-like): The true labels.

        Returns:
        - matrix (array): Confusion matrix.
        """
        y_pred = self.predict(X)
        return confusion_matrix(y_true, y_pred)