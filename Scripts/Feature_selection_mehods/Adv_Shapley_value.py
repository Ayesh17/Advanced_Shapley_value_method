import pandas as pd
import shap
import numpy as np
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Define the function to calculate shapley value for each feature
def shapley_value_calc(X, y, feature):
    # Create a background dataset to represent the distribution of the data
    background_data = shap.maskers.Independent(X[:100], max_samples=100)

    # Define the model and fit it to the data
    model = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42)
    model.fit(X.astype(np.float64), y)

    # Create an explainer object using a specific model
    explainer = shap.Explainer(model, background_data)

    # Compute the Shapley values for the feature of interest
    shap_values = explainer(X, check_additivity=False)[:, feature]
    print("shap", shap_values.values)
    print("shap", shap_values.shape)

    # Extract the values from the Explanation object and calculate the mean Shapley value
    mean_shap = np.mean(np.abs(shap_values.values))
    print("mean", mean_shap)

    # Calculate the shapley value using np.ndarray.min()
    shapley_value = mean_shap - np.array(shap_values.values).min()

    # Print the result
    print("The shapley value for feature {} is: {:.4f}".format(feature, shapley_value))

    return shapley_value



def get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols):
    # Calculate the shapley value for each feature
    shapley_value = [shapley_value_calc(X_train, y_train, i) for i in range(X_train.shape[1])]
    print("shapley_val", shapley_value)

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
