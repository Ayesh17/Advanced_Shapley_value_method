from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import shap
import random


# Load the data from the CSV file
# df = pd.read_csv('haberman.csv')
df = pd.read_csv('iris_dataset.csv')

# Get the x and y values as NumPy arrays
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(X.shape)
print(y.shape)

# Define the number of features to select
num_features = 3

# Define the threshold for the Nucleus method
nucleus_threshold = 0.95

# Compute the Gini importance of each feature
rfc = RandomForestClassifier()
rfc.fit(X, y)
gini_importance = rfc.feature_importances_

# Compute the Shapley values of each feature
explainer = shap.TreeExplainer(rfc)
shap_values = explainer.shap_values(X)
shap_values = np.abs(shap_values).mean(axis=0)
print("shap",shap_values.shape)

# Combine the Gini and Shapley values using a weighted sum
weights = [0.5, 0.5]
gini_shap = (weights[0] * gini_importance) + (weights[1] * shap_values)

# Use the Nucleus method to select the top features
sorted_indices = np.argsort(gini_shap)[::-1]
cumulative_probs = np.cumsum(gini_shap[sorted_indices])
num_features_selected = 0
for i in range(len(cumulative_probs)):
    if cumulative_probs[i] >= nucleus_threshold:
        num_features_selected = i + 1
        break
selected_indices = sorted_indices[:num_features_selected]
print("ind",selected_indices)


# Train a new random forest using only the selected features
X_selected = X[:, selected_indices]
print("select",X_selected.shape)
X_selected = X_selected.reshape(-1, num_features_selected)  # Reshape the input data to have only two dimensions
print("after",X_selected.shape)
y_selected = y  # Select only the corresponding target values
rfc_selected = RandomForestClassifier()
rfc_selected.fit(X_selected, y_selected)

# Compute the accuracy of the new random forest on the training data
y_pred = rfc_selected.predict(X_selected)
accuracy = accuracy_score(y_selected, y_pred)
print(f"Accuracy: {accuracy:.2f}")
