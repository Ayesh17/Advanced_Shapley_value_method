import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from shapely import geometry
from sklearn.datasets import load_wine


# Load the data from the CSV file
# df = pd.read_csv('boston_housing.csv')
df = pd.read_csv('haberman.csv')
# df = pd.read_csv('iris_dataset.csv')

# Get the x and y values as NumPy arrays
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Ensure X is a 2D numpy array and y is a 1D numpy array
X = np.atleast_2d(X)
y = np.squeeze(y)

# Print the shape of x and y to verify they are correct
print('x shape:', X.shape)
print('y shape:', y.shape)

# Load your data and split it into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the function to calculate Nucleus for each feature
def nucleus(X, y, feature):
    # Calculate the class distribution for the feature subset
    pos_count = sum(y[X[:, feature] == 1] == 1)
    neg_count = sum(y[X[:, feature] == 1] == 0)
    subset_count = pos_count + neg_count

    # If the subset is empty, return 0
    if subset_count == 0:
        return 0

    # Calculate the proportion of positive samples in the subset
    subset_pos_ratio = pos_count / subset_count

    # Calculate the proportion of the subset in the total dataset
    subset_proportion = subset_count / len(X)

    # Calculate the Nucleus for the feature
    return subset_proportion * np.log2(subset_pos_ratio)


# Define the function to calculate shapley value for each feature
def shapley_value(X, y, feature):
    # Create two subsets of the data
    X1 = pd.DataFrame(X)[pd.DataFrame(X).iloc[:, feature] == 0]
    X2 = X[X[:, feature] == 1]

    # Check if X1 is empty
    if len(X1) == 0:
        return 0

    # Calculate the proportion of each subset in the total dataset
    p1 = len(X1) / len(X)
    p2 = len(X2) / len(X)

    # Calculate the probabilities of the positive class in each subset
    p1_pos = sum(y[X[:, feature] == 0] == 1) / len(X1)
    p2_pos = sum(y[X[:, feature] == 1] == 1) / len(X2)

    # Calculate the expected positive class probability for each subset
    e1_pos = p1_pos * p1 + p2_pos * p2
    e2_pos = p1_pos * p2 + p2_pos * p1

    # Calculate the Shapley value for the feature
    return abs(e1_pos - e2_pos)

# Calculate the Nucleus for each feature
nucleus_val = [nucleus(X_train, y_train, i) for i in range(X_train.shape[1])]

# Calculate the shapley value for each feature
shapley_val = [shapley_value(X_train, y_train, i) for i in range(X_train.shape[1])]

# Combine the Nucleus and shapley values for each feature
combined_values = [nucleus_val[i] * shapley_val[i] for i in range(X_train.shape[1])]

# # Select the most important feature using the combined values
# most_important_feature = np.argmax(combined_values)
# # Create a subset of the data with only the most important feature
# X_train_reduced = X_train[:, most_important_feature].reshape(-1, 1)
# X_test_reduced = X_test[:, most_important_feature].reshape(-1, 1)

# Select the most important feature using the Nucleus values
least_important_feature_shap = np.argmin(shapley_value)
print("least_imp_shap",least_important_feature_shap)

# Select the most important feature using the Nucleus values
least_important_feature_nuc = np.argmin(nucleus_val)
print("least_imp_nuc",least_important_feature_nuc)

# Create a subset of the data with only the most important feature
mask = np.ones(X.shape[1], dtype=bool)
mask[least_important_feature_shap] = False
X_train_reduced = X_train[:, mask]
X_test_reduced = X_test[:, mask]

# Create a subset of the data with only the most important feature
if least_important_feature_shap!=least_important_feature_nuc:
    mask = np.ones(X.shape[1], dtype=bool)
    mask[least_important_feature_nuc] = False
    X_train_reduced = X_train[:, mask]
    X_test_reduced = X_test[:, mask]




# Define the random forest with three decision trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the random forest to the reduced training data
rf.fit(X_train_reduced, y_train)

# Evaluate the accuracy of the random forest on the reduced testing data
y_pred = rf.predict(X_test_reduced)
print(y_test)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
