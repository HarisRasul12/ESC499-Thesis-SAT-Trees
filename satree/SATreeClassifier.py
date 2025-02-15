"""
=========== Module Description ===========

SAT Tree model classifier. This module provides a classifier that uses a pre-built decision tree to make predictions
and evaluate performance. The tree is based on the SAT solution for the training dataset.
"""

from typing import List, Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class SATreeClassifier:

    def __init__(self, tree: List[Dict]) -> None:
        """
        Initializes the SATreeClassifier with a pre-built decision tree model derived from a SAT solution.

        Args:
            tree: A list of dictionaries representing the decision tree structure.
                  Each dictionary corresponds to a node in the tree and includes keys such as 'type', 'feature',
                  'threshold', 'children', and, for leaf nodes, 'label'.
        """
        self.tree_model = tree

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for the given data using the SAT-based decision tree.

        Args:
            data: A numpy array of input samples where each row represents a single data point.

        Returns:
            A numpy array containing the predicted labels for each input sample.
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

    def score(self, dataset: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the accuracy of the classifier on the provided dataset.

        Args:
            dataset: A numpy array of input features for which predictions are made.
            y_true: A numpy array of true labels corresponding to the dataset.

        Returns:
            A float representing the accuracy of the model.
        """
        y_pred = self.predict(dataset)
        return accuracy_score(y_true, y_pred)

    def get_classification_report(self, dataset: np.ndarray, y_true: np.ndarray) -> str:
        """
        Generates a classification report summarizing precision, recall, and F1 scores for the classifier's predictions.

        Args:
            dataset: A numpy array of input features for which predictions are made.
            y_true: A numpy array of true labels corresponding to the dataset.

        Returns:
            A string containing the classification report.
        """
        y_pred = self.predict(dataset)
        return classification_report(y_true, y_pred)

    def get_confusion_matrix(self, dataset: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Computes the confusion matrix for the classifier's predictions.

        Args:
            dataset: A numpy array of input features for which predictions are made.
            y_true: A numpy array of true labels corresponding to the dataset.

        Returns:
            A numpy array representing the confusion matrix.
        """
        y_pred = self.predict(dataset)
        return confusion_matrix(y_true, y_pred)
