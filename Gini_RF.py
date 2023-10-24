# Load the data from the CSV file
import pandas as pd
import numpy as np

# df = pd.read_csv('haberman.csv')
# df = pd.read_csv('haberman_edited.csv')
# df = pd.read_csv('iris_dataset.csv')
df = pd.read_csv('diabetes.csv')

# Get the x and y values as NumPy arrays
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Ensure X is a 2D numpy array and y is a 1D numpy array
X = np.atleast_2d(X)
y = np.squeeze(y)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model on the testing set
from sklearn.metrics import accuracy_score

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
