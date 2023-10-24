from itertools import combinations

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Define the function to calculate Nucleus for each feature
import numpy as np
from nashpy import Game

def calculate_nucleolus(X, y, feature):
    # Check if the feature is a valid index
    if feature not in range(1, X.shape[1] + 1):
        raise ValueError("Feature must be an index into the list of features.")

    # Create a list of all possible coalitions of features
    features = [i for i in range(1, X.shape[1] + 1) if i != feature]
    coalitions = [[]] + [list(comb) for r in range(len(features)) for comb in combinations(features, r + 1)]

    # Calculate the payoff for each coalition
    payoffs = []
    for coalition in coalitions:
        # Create a matrix of data for the given coalition
        X_c = X[:, coalition]
        # Fit a linear model to the data
        beta = np.linalg.lstsq(X_c, y, rcond=None)[0]
        # Calculate the payoff for the coalition
        payoff = np.dot(X_c, beta)
        payoffs.append(payoff)

    # Create a game object with the payoffs
    game = Game(payoffs)

    # Calculate the nucleolus value for the given feature
    feature_index = features.index(feature)
    nucleolus = game.nucleolus()[feature_index]

    return nucleolus



def get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols):
    # Initialize the nucleolus values
    nucleolus_values = []

    # Calculate the nucleolus value for each feature
    for i in range(X_train.shape[1]):
        nucleolus_values.append(calculate_nucleolus(X_train, y_train, i))

    # Find the features with the lowest nucleolus values
    best_features = [i for i, value in enumerate(nucleolus_values) if value == min(nucleolus_values)]

    # Reduce the training and test data to only include the best features
    X_train_reduced = X_train[:, best_features]
    X_test_reduced = X_test[:, best_features]

    return X_train_reduced, X_test_reduced
