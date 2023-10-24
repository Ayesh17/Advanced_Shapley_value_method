import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from shapely import geometry


# Define the function to calculate Nucleus for each feature
def nucleus(X, y, feature):
    # Map the string labels to integers
    y = y.astype(int)

    # Get the unique values in the feature subset
    values = np.unique(X[:, feature])

    # Calculate the class distribution for each value in the feature subset
    subset_counts = {}
    for value in values:
        subset_counts[value] = sum(y[X[:, feature] == value])

    # If the subset is empty, return 0
    if sum(subset_counts.values()) == 0:
        return 0

    # Calculate the entropy of the class distribution for the subset
    subset_entropy = 0
    subset_count = sum(subset_counts.values())
    for class_label in set(y):
        class_indices = np.where(y[X[:, feature] == class_label])[0]
        class_ratio = len(class_indices) / subset_count
        if class_ratio > 0:
            subset_entropy -= class_ratio * np.log2(class_ratio)

    # Calculate the proportion of the subset in the total dataset
    subset_proportion = subset_count / len(X)

    # Calculate the Nucleus for the feature
    return subset_proportion * subset_entropy



# Define the function to calculate Shapley value for each feature
def shapley_value_calc(X, y, feature):
    # Convert input to NumPy array
    X = np.array(X)

    # Get the median value of the feature
    median_val = np.median(X[:, feature])
    # print("Median_val",median_val)

    # Create two subsets of the data based on feature value
    X1 = X[X[:, feature] < median_val]
    X2 = X[X[:, feature] >= median_val]

    # Check if X1 is empty
    if len(X1) == 0:
        return 0

    # Calculate the proportion of each subset in the total dataset
    p1 = len(X1) / len(X)
    p2 = len(X2) / len(X)

    # Calculate the probabilities of the positive class in each subset
    p1_pos = sum(y[X[:, feature] < median_val] == 1) / len(X1)
    p2_pos = sum(y[X[:, feature] >= median_val] == 1) / len(X2) if len(X2) > 0 else 0

    # Calculate the expected positive class probability for each subset
    e1_pos = p1_pos * p1 + p2_pos * p2
    e2_pos = p1_pos * p2 + p2_pos * p1

    # Calculate the Shapley value for the feature
    return abs(e1_pos - e2_pos)


def get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols):
    # Calculate the Nucleus for each feature
    nucleus_values = [nucleus(X_train, y_train, i) for i in range(X_train.shape[1])]
    print("nuc_val",nucleus_values)

    # Calculate the shapley value for each feature
    shapley_value = [shapley_value_calc(X_train, y_train, i) for i in range(X_train.shape[1])]
    print("shap_val",shapley_value)

    # Combine the Nucleus and shapley values for each feature
    combined_values = [nucleus_values[i] * shapley_value[i] for i in range(X_train.shape[1])]

    # Determine the number of features to select
    num_features_to_select = int(num_cols* 0.3)
    print("num_features_to_select", num_features_to_select)

    # Sort the Shapley values in ascending order
    sorted_shapley_values = np.argsort(shapley_value)
    # Select the shapley least important features
    least_important_features_shap = sorted_shapley_values[:num_features_to_select]
    print("least_imp_shap",least_important_features_shap)


    # Sort the Nucleus values in ascending order
    sorted_nucleus_values = np.argsort(nucleus_values)
    # Select the nucleus least important features
    least_important_features_nuc = sorted_nucleus_values[:num_features_to_select]
    print("least_imp_nuc",least_important_features_nuc)

    # Select the combined least important features
    least_important_features_shap_set = set(least_important_features_shap)
    least_important_features_nuc_set = set(least_important_features_nuc)
    combined_indices = list(least_important_features_shap_set.union(least_important_features_nuc_set))

    print()
    print("combined_indices",combined_indices)

    # Remove the least important features from the data
    mask = np.ones(X.shape[1], dtype=bool)
    mask[combined_indices[:int(len(combined_indices) * 0.35)]] = False
    X_train_reduced = X_train[:, mask]
    X_test_reduced = X_test[:, mask]

    return X_train_reduced, X_test_reduced


