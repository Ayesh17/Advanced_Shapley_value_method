import pandas as pd
import shap
import numpy as np
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Define the function to calculate nucleolus value for each feature
def nucleolus_value_calc(X, y, feature):
    # Create a background dataset to represent the distribution of the data
    background_data = shap.maskers.Independent(X[:100], max_samples=100)

    # Define the model and fit it to the data
    model = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42)
    model.fit(X, y)

    # Create an explainer object using a specific model
    explainer = shap.Explainer(model, background_data)

    # Compute the Shapley values for the feature of interest
    shap_values = explainer(X, check_additivity=False)[:, feature]
    print("shap", shap_values)

    # Extract the values from the Explanation object and calculate the mean Shapley value
    mean_shap = np.mean(np.abs(shap_values.values))
    print("mean", mean_shap)

    # Calculate the nucleolus value using np.ndarray.min()
    nucleolus_value = mean_shap - np.array(shap_values.values).min()

    # Print the result
    print("The nucleolus value for feature {} is: {:.4f}".format(feature, nucleolus_value))

    return nucleolus_value



def get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols):
    # Calculate the nucleolus value for each feature
    nucleolus_value = [nucleolus_value_calc(X_train, y_train, i) for i in range(X_train.shape[1])]
    print("nucleolus_val", nucleolus_value)

    # Sort the features in descending order of importance
    sorted_nucleolus_values = np.argsort(nucleolus_value)[::-1]


    # Determine the number of features to keep by selecting the top 70% of the most important features
    num_features = int(np.ceil(0.7 * len(X_train[0])))

    # Select the top num_features features to keep
    top_indices = sorted_nucleolus_values[:num_features]

    # Create new feature sets X_train_reduced and X_test_reduced by selecting the top features
    X_train_reduced = X_train[:, top_indices]
    X_test_reduced = X_test[:, top_indices]

    return X_train_reduced, X_test_reduced
