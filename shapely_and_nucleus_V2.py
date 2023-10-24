import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from shapely import geometry

# Load the data from the CSV file
# df = pd.read_csv('cat_dog_data.csv')
# df = pd.read_csv('haberman.csv')
# df = pd.read_csv('haberman_edited.csv')
df = pd.read_csv('diabetes.csv')

# df = pd.read_csv('iris_dataset.csv')
# df['species'].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2], inplace=True)

# df = pd.read_csv('adult.csv')
# # df = df.drop(columns=['fnlwgt', 'education'])  # drop unnecessary columns
# df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)  # convert income to binary label
# df = pd.get_dummies(df)  # one-hot encode categorical variables
# print(df.shape)
# X = df.drop(columns=['income'])
# y = df['income']

# df = pd.read_csv('bank.csv')
# df['deposit'] = df['deposit'].apply(lambda x: 1 if x.strip() == 'yes' else 0)  # convert income to binary label
# df = pd.get_dummies(df)  # one-hot encode categorical variables
# print(df.shape)
# X = df.drop(columns=['deposit'])
# y = df['deposit']





# # Get the x and y values as NumPy arrays
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


num_cols = df.shape[1]

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
        class_indices = np.where(y[X[:, feature] == value] == class_label)[0]
        class_ratio = len(class_indices) / subset_count
        if class_ratio > 0:
            subset_entropy -= class_ratio * np.log2(class_ratio)

    # Calculate the proportion of the subset in the total dataset
    subset_proportion = subset_count / len(X)

    # Calculate the Nucleus for the feature
    return subset_proportion * subset_entropy



# Define the function to calculate Shapley value for each feature
def shapley_value(X, y, feature):
    # Convert input to NumPy array
    X = np.array(X)

    # Get the median value of the feature
    print("med",X[:, feature])
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



# Calculate the Nucleus for each feature
nucleus_values = [nucleus(X_train, y_train, i) for i in range(X_train.shape[1])]
print("nuc_val",nucleus_values)

# Calculate the shapley value for each feature
shapley_value = [shapley_value(X_train, y_train, i) for i in range(X_train.shape[1])]
print("shap_val",shapley_value)

# Combine the Nucleus and shapley values for each feature
combined_values = [nucleus_values[i] * shapley_value[i] for i in range(X_train.shape[1])]

# # Select the most important feature using the combined values
# most_important_feature = np.argmax(combined_values)
# # Create a subset of the data with only the most important feature
# X_train_reduced = X_train[:, most_important_feature].reshape(-1, 1)
# X_test_reduced = X_test[:, most_important_feature].reshape(-1, 1)

# # Select the most important feature using the Nucleus values
# least_important_feature_shap = np.argmin(shapley_value)

# Determine the number of features to select
num_features_to_select = int(num_cols* 0.3)
print(num_features_to_select)

# Sort the Shapley values in ascending order
sorted_shapley_values = np.argsort(shapley_value)
# Select the `num_features_to_select` least important features
least_important_features_shap = sorted_shapley_values[:num_features_to_select]
print("least_imp_shap",least_important_features_shap)



# Determine the number of features to select
# Sort the Shapley values in ascending order
sorted_nucleus_values = np.argsort(nucleus_values)
# Select the `num_features_to_select` least important features
least_important_features_nuc = sorted_nucleus_values[:num_features_to_select]
print("least_imp_nuc",least_important_features_nuc)

least_important_features_shap_set = set(least_important_features_shap)
least_important_features_nuc_set = set(least_important_features_nuc)
combined_indices = list(least_important_features_shap_set.union(least_important_features_nuc_set))
print()
print("combined_indices",combined_indices)


# # Create a subset of the data with only the most important feature
# mask = np.ones(X.shape[1], dtype=bool)
# mask[least_important_feature_shap] = False
# X_train_reduced = X_train[:, mask]
# X_test_reduced = X_test[:, mask]

# Remove the least important features from the data
mask = np.ones(X.shape[1], dtype=bool)
mask[combined_indices[:int(len(combined_indices)*0.35)]] = False
X_train_reduced = X_train[:, mask]
X_test_reduced = X_test[:, mask]


# # Create a subset of the data with only the most important feature
# if least_important_feature_shap!=least_important_feature_nuc:
#     mask = np.ones(X.shape[1], dtype=bool)
#     mask[least_important_feature_nuc] = False
#     X_train_reduced = X_train[:, mask]
#     X_test_reduced = X_test[:, mask]




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







print()
print()
print()
print()
print("Normal RF")

# # # Get the x and y values as NumPy arrays
# # X = df.iloc[:, :-1].values
# # y = df.iloc[:, -1].values
# #
# # # Ensure X is a 2D numpy array and y is a 1D numpy array
# # X = np.atleast_2d(X)
# # y = np.squeeze(y)
#
# # Split the data into training and testing sets
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train a random forest classifier
# from sklearn.ensemble import RandomForestClassifier
#
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)
#
# # Evaluate the model on the testing set
# from sklearn.metrics import accuracy_score
#
# y_pred = rf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
#
# print(f"Accuracy: {accuracy}")


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('adult.csv')
# Preprocess data
# df = df.drop(columns=['fnlwgt', 'education'])  # drop unnecessary columns
df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)  # convert income to binary label
df = pd.get_dummies(df)  # one-hot encode categorical variables
print(df.shape)
X = df.drop(columns=['income'])
y = df['income']

# df = pd.read_csv('bank.csv')
# df['deposit'] = df['deposit'].apply(lambda x: 1 if x.strip() == 'yes' else 0)  # convert income to binary label
# df = pd.get_dummies(df)  # one-hot encode categorical variables
# print(df.shape)
# X = df.drop(columns=['deposit'])
# y = df['deposit']



# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the performance of the classifier
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
