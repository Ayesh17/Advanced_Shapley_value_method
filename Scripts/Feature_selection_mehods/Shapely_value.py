import pandas as pd
import numpy as np
import itertools
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Define the function to calculate Shapley value for each feature
# def shapley_value_calc(X, y, feature):
#     # Convert input to NumPy array
#     X = np.array(X)
#
#     # Get the number of features
#     num_features = X.shape[1]
#
#     # Initialize the Shapley value to zero
#     shapley_value = 0.0
#
#
#     # Iterate over all possible permutations of features
#     for perm in itertools.permutations(range(num_features)):
#         # print("perm", perm)
#         # Find the index of the current feature in the permutation
#         index = perm.index(feature)
#         # print("index",index)
#
#         # Check if the current feature is the first feature in the permutation
#         if index == 0:
#             # Calculate the contribution of the first feature to the prediction
#             X_perm = X[:, perm]
#             p = np.mean(y[X_perm[:, index] <= np.median(X_perm[:, index])])
#             shapley_value += p
#
#         # Otherwise, calculate the contribution of the current feature to the prediction
#         else:
#             # Calculate the contribution of all features before the current feature
#             X_perm = X[:, perm[:index]]
#             p1 = np.mean(y[X_perm[:, index-1] <= np.median(X_perm[:, index-1])])
#             X_perm = X[:, perm[:index+1]]
#             p2 = np.mean(y[(X_perm[:, index-1] <= np.median(X_perm[:, index-1])) & (X_perm[:, index] >= np.median(X_perm[:, index]))])
#             shapley_value += (p2 - p1) * math.factorial(index) * math.factorial(num_features - index - 1)
#
#     # Normalize the Shapley value
#     print()
#     print("shapley val1", shapley_value)
#     shapley_value /= math.factorial(num_features)
#
#     print()
#     print("shapley val",shapley_value)
#     return shapley_value


def shapley_value_calc(X, y, feature):
    # Convert input to NumPy array
    X = np.array(X)

    # Get the number of features
    num_features = X.shape[1]

    # Initialize the Shapley value to zero
    shapley_values = []


    # Iterate over all possible permutations of features
    for perm in itertools.permutations(range(num_features)):
        # print("perm", perm)
        # Find the index of the current feature in the permutation
        index = perm.index(feature)
        # print("index",index)

        # Check if the current feature is the first feature in the permutation
        if index == 0:
            # Calculate the contribution of the first feature to the prediction
            X_perm = X[:, perm]
            p = np.mean(y[X_perm[:, index] <= np.median(X_perm[:, index])])
            shapley_values.append(p)

        # Otherwise, calculate the contribution of the current feature to the prediction
        else:
            # Calculate the contribution of all features before the current feature
            X_perm = X[:, perm[:index]]
            p1 = np.mean(y[X_perm[:, index-1] <= np.median(X_perm[:, index-1])])
            X_perm = X[:, perm[:index+1]]
            p2 = np.mean(y[(X_perm[:, index-1] <= np.median(X_perm[:, index-1])) & (X_perm[:, index] >= np.median(X_perm[:, index]))])
            shapley_values.append((p2 - p1) * math.factorial(index) * math.factorial(num_features - index - 1))

    # Normalize the Shapley value
    print()
    print("shapley val1", shapley_values)
    shapley_value = np.mean(shapley_values)
    shapley_value = shapley_value - np.array(shapley_values).min()

    print()
    print("shapley val",shapley_value)
    return shapley_value



# def get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols):
#
#     # Calculate the shapley value for each feature
#     shapley_value = [shapley_value_calc(X_train, y_train, i) for i in range(X_train.shape[1])]
#     print("shap_val",shapley_value)
#
#
#     # Determine the number of features to select
#     num_features_to_select = int(num_cols* 0.3)
#     print("num_features_to_select", num_features_to_select)
#
#     # Sort the Shapley values in ascending order
#     sorted_shapley_values = np.argsort(shapley_value)
#     # Select the shapley least important features
#     least_important_features_shap = sorted_shapley_values[:num_features_to_select]
#     print("least_imp_shap",least_important_features_shap)
#
#
#
#     # Select the combined least important features
#     least_important_features_shap_set = set(least_important_features_shap)
#     shap_indices = list(least_important_features_shap_set)
#
#     print()
#     print("shap_indices",shap_indices)
#
#     # Remove the least important features from the data
#     mask = np.ones(X.shape[1], dtype=bool)
#     mask[shap_indices[:int(len(shap_indices) * 0.35)]] = False
#     X_train_reduced = X_train[:, mask]
#     X_test_reduced = X_test[:, mask]
#
#     return X_train_reduced, X_test_reduced



def get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols):
    # Calculate the shapley value for each feature
    shapley_value = [shapley_value_calc(X_train, y_train, i) for i in range(X_train.shape[1])]
    print("shap_val", shapley_value)

    # Sort the features in descending order of importance
    sorted_shapley_values = np.argsort(shapley_value)[::-1]


    # Determine the number of features to keep by selecting the top 70% of the most important features
    num_features = int(np.ceil(0.7 * len(X_train[0])))

    # Select the top num_features features to keep
    top_indices = sorted_shapley_values[:num_features]

    # Create new feature sets X_train_reduced and X_test_reduced by selecting the top features
    X_train_reduced = X_train[:, top_indices]
    X_test_reduced = X_test[:, top_indices]

    return X_train_reduced, X_test_reduced