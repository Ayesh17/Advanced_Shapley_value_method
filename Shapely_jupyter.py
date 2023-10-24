import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# Load the data using pandas
file_name = "haberman.csv"
# # file_name = "haberman_edited.csv"
# file_name = "iris_dataset.csv"

data = pd.read_csv(file_name)
if (file_name == "iris_dataset.csv"):
    data['species'].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                            [0, 1, 2], inplace=True)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the dataset into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Print the sizes of the resulting datasets
print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Testing set size:", len(X_test))

# Prepares a default instance of the random forest classifier
model = RandomForestClassifier()
# Fits the model on the data
model.fit(X_train, y_train)

# Fits the explainer
explainer = shap.Explainer(model.predict, X_val)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_val)

print(shap_values.values[1])

# Use the reduced SHAP values to plot feature importances
shap.summary_plot(shap_values, X_val, plot_type="bar")

# Calculate the average absolute SHAP value for each feature
avg_shap_values = np.abs(shap_values.values).mean(axis=0)

# Get the indices of the lowest 30% of features based on their average absolute SHAP value
print(len(avg_shap_values))
n_features_to_remove = int(0.35 * len(avg_shap_values))
print("no",n_features_to_remove)
lowest_avg_shap_indices = avg_shap_values.argsort()[:n_features_to_remove]

print(lowest_avg_shap_indices)

X_train = np.array(X_train)

print("X_train",X_train.shape)
# Remove the lowest 30% of features from the test set and re-calculate SHAP values
X_train_reduced = np.delete(X_train, lowest_avg_shap_indices, axis=1)
print("X_train_reduced",X_train_reduced.shape)

# Define the random forest with three decision trees
rf = RandomForestClassifier(n_estimators=50, random_state=42)

# Fit the random forest to the reduced training data
rf.fit(X_train_reduced, y_train)

X_test = np.array(X_test)

print("X_test",X_test.shape)
# Remove the lowest 30% of features from the test set and re-calculate SHAP values
X_test_reduced = np.delete(X_test, lowest_avg_shap_indices, axis=1)
print("X_test_reduced",X_test_reduced.shape)

# Evaluate the accuracy of the random forest on the reduced testing data
y_pred = rf.predict(X_test_reduced)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
