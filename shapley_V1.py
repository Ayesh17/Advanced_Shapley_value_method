import pandas as pd
import numpy as np
import shap
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('iris_dataset.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Calculate Shapley values
explainer = shap.Explainer(DecisionTreeClassifier().fit(X, y).predict_proba, X)
shap_values = explainer(X)

# Sum up the Shapley values across all samples to get the feature importance
feature_importance = np.abs(shap_values.values).mean(0)

# Sort features by their importance and select the top K features
k = 2
top_feature_indices = np.argsort(feature_importance)[-k:]
top_features = df.columns[top_feature_indices]

# Create a decision tree using only the top K features
clf = DecisionTreeClassifier()
clf.fit(np.take(X, top_feature_indices, axis=1), y)

# Make predictions on new data
X_new = pd.read_csv("iris_dataset.csv")
predictions = clf.predict(np.take(X_new, top_feature_indices, axis=1))
