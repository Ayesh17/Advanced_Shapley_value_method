from itertools import combinations

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Define the function to calculate Nucleus for each feature
def calculate_nucleolus(X, y, feature):
    # Create a list of all possible coalitions of features
    features = [i for i in range(X.shape[1]) if i != feature]
    coalitions = [[]] + [list(comb) for r in range(len(features)) for comb in combinations(features, r + 1)]

    # Calculate the payoff for each coalition
    payoffs = []
    for coalition in coalitions:
        print()
        print("coalition ", coalition)
        X_c = X[:, coalition]
        X_c = np.concatenate((X_c, X[:, feature][:, None]), axis=1)
        beta = np.linalg.lstsq(X_c, y, rcond=None)[0]
        payoff = np.dot(X_c, beta)
        payoffs.append(payoff)

    # Calculate the nucleolus value
    nucleolus = np.zeros(len(coalitions))
    for i, coalition in enumerate(coalitions):
        deviation = payoffs[i] - np.mean(payoffs)
        nucleolus[i] = np.sum(np.minimum(0, deviation)) / len(coalition)

    return nucleolus[-1]



def get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols):
    # Calculate the Nucleus for each feature
    nucleus_values = [calculate_nucleolus(X_train, y_train, i) for i in range(X_train.shape[1])]
    print("nuc_val",nucleus_values)

    # Determine the number of features to select
    num_features_to_select = int(num_cols* 0.3)
    print("num_features_to_select", num_features_to_select)


    # Sort the Nucleus values in ascending order
    sorted_nucleus_values = np.argsort(nucleus_values)
    # Select the nucleus least important features
    least_important_features_nuc = sorted_nucleus_values[:num_features_to_select]
    print("least_imp_nuc",least_important_features_nuc)

    # Select the combined least important features
    least_important_features_nuc_set = set(least_important_features_nuc)
    nucleus_indices = list(least_important_features_nuc_set)

    print()
    print("nucleus_indices",nucleus_indices)

    # Remove the least important features from the data
    mask = np.ones(X.shape[1], dtype=bool)
    mask[nucleus_indices[:int(len(nucleus_indices) * 0.35)]] = False
    X_train_reduced = X_train[:, mask]
    X_test_reduced = X_test[:, mask]

    return X_train_reduced, X_test_reduced
