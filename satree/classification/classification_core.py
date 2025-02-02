import numpy as np


# Helper function to sort data points by feature and create O_j FOR CATGEORICAL
def compute_ordering_with_categorical(X, feature_index, features_categorical):
    # Determine if the current feature is categorical
    is_categorical = str(feature_index) in features_categorical

    if is_categorical:
        # Group identical categories together and maintain their index order
        unique_categories = np.unique(X[:, feature_index])
        ordering = sum((list(np.where(X[:, feature_index] == category)[0])
                        for category in unique_categories), [])
    else:
        # For numerical features, convert to float then sort by feature value
        numerical_values = X[:, feature_index].astype(float)
        ordering = np.argsort(numerical_values).tolist()

    return ordering
