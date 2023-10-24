import pandas as pd
import shap
import numpy as np
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Define the function to calculate nucleolus value for each feature
def shapley_value_calc(X, y, feature):
    print("feature", feature)
    # Create a background dataset to represent the distribution of the data
    background_data = shap.maskers.Independent(X[:100], max_samples=100)

    # Define the model and fit it to the data
    model = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42)
    model.fit(X, y)

    # Create an explainer object using a specific model
    explainer = shap.Explainer(model, background_data)

    # Compute the Shapley values for the feature of interest
    shap_values = explainer(X, check_additivity=False)
    # print("shap1 size:", len(shap_values.values))
    # print("shap", shap_values.values[0])
    # print()
    shap_values = explainer(X, check_additivity=False)[:, feature]
    print()
    print("shap2 size:", len(shap_values.values))
    print("shap", shap_values.values)


    # shap.plots.waterfall(shap_values[0])
    # Extract the values from the Explanation object and calculate the mean Shapley value
    mean_shap = np.mean(np.abs(shap_values.values))
    print("mean_shap", mean_shap)

    # Normalize the Shapley value
    num_features = X.shape[1]
    shapley_value = mean_shap / num_features
    print("shapley_value", shapley_value)


    # Return the Shapley value "explain this step by step"

    return shapley_value



def get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols):
    # Calculate the nucleolus value for each feature
    nucleolus_value = [shapley_value_calc(X_train, y_train, i) for i in range(X_train.shape[1])]
    # print("nucleolus_val", nucleolus_value)

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
