import math

import pandas as pd
import shap
import numpy as np
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Define the function to calculate nucleolus value for each feature
import gambit
import gambit
import numpy as np
import nashpy as nash
import itertools

def nucleolus_value_calc(X, y, feature):
    # Convert input to NumPy array
    X = np.array(X)

    # Get the number of features
    num_features = X.shape[1]

    # Initialize the nucleolus value to zero
    nucleolus_value = 0.0

    # Iterate over all possible permutations of features
    for perm in itertools.permutations(range(num_features)):
        print("perm", perm)
        # Find the index of the current feature in the permutation
        index = perm.index(feature)
        print("index", index)

        # Check if the current feature is the first feature in the permutation
        if index == 0:
            # Calculate the contribution of the first feature to the prediction
            X_perm = X[:, perm]
            p = np.mean(y[X_perm[:, index] <= np.median(X_perm[:, index])])
            nucleolus_value += p

        # Otherwise, calculate the contribution of the current feature to the prediction
        else:
            # Calculate the contribution of all features before the current feature
            X_perm = X[:, perm[:index]]
            p1 = np.mean(y[X_perm[:, index-1] <= np.median(X_perm[:, index-1])])
            X_perm = X[:, perm[:index+1]]
            p2 = np.mean(y[(X_perm[:, index-1] <= np.median(X_perm[:, index-1])) & (X_perm[:, index] >= np.median(X_perm[:, index]))])
            nucleolus_value += (p2 - p1) * math.factorial(index) * math.factorial(num_features - index - 1)

    # Normalize the nucleolus value
    nucleolus_value /= math.factorial(num_features)

    print()
    print("nucleolus val", nucleolus_value)
    return nucleolus_value



def get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols):
    # Calculate the nucleolus value for each feature
    nucleolus_value = [nucleolus_value_calc(X_train, y_train, i) for i in range(X_train.shape[1])]
    print("nucleolus_val", nucleolus_value)

    # Sort the features in descending order of importance
    sorted_nucleolus_values = np.argsort(nucleolus_value)[::1]


    # Determine the number of features to keep by selecting the top 70% of the most important features
    num_features = int(np.ceil(0.7 * len(X_train[0])))

    # Select the top num_features features to keep
    top_indices = sorted_nucleolus_values[:num_features]

    # Create new feature sets X_train_reduced and X_test_reduced by selecting the top features
    X_train_reduced = X_train[:, top_indices]
    X_test_reduced = X_test[:, top_indices]

    return X_train_reduced, X_test_reduced
