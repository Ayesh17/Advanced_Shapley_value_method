from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import shap

# Load the data from the CSV file
df = pd.read_csv('iris_dataset.csv')


# Get the x and y values as NumPy arrays
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Compute the Shapley values of each feature
rfc = RandomForestClassifier()
rfc.fit(X, y)
explainer = shap.TreeExplainer(rfc)
print("x",X.shape)
print("y",y.shape)
shap_values = np.abs(explainer.shap_values(X)).mean(axis=0)
print("s1",shap_values.shape)
shap_values /= shap_values.sum()  # normalize to sum up to 1
print("s2",shap_values.shape)

# Train a new random forest using the Shapley values as sample weights
rfc_shap = RandomForestClassifier()
rfc_shap.fit(X, y, sample_weight=shap_values)

# Compute the accuracy of the new random forest on the training data
y_pred = rfc_shap.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.2f}")
