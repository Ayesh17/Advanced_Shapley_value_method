import itertools
import math
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Sample dataset
X = [[0, 0], [1, 1], [0, 1], [1, 0]]
y = [0, 0, 1, 1]

# Load your data and split it into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the function to calculate Shapley value for each feature
def shapely_value(matrix, coalition):
    # Determine the worth of the coalition by computing the sum of values in the matrix for the players in the coalition
    worth = sum(matrix[i][j] for i in coalition for j in range(len(matrix[i])))

    # Determine the number of players in the coalition
    num_players = len(coalition)

    # Compute the Shapely value as the average marginal contribution over all possible orders
    marginal_contributions = []
    for i in coalition:
        for order in itertools.permutations(coalition - {i}):
            contribution = sum(matrix[i][j] for j in order) - worth
            marginal_contributions.append(contribution / math.factorial(num_players - 1))

    return sum(marginal_contributions) / num_players

def decision_tree(X, y, matrix, players):
    if len(set(y)) == 1:
        return y[0]

    best_feature, best_threshold, max_shapely = None, None, -float('inf')

    # Iterate over all possible splits and compute the Shapely value for each
    for feature in range(len(X[0])):
        for threshold in [0, 1]:
            left_indices = [i for i in range(len(X)) if X[i][feature] <= threshold]
            right_indices = [i for i in range(len(X)) if X[i][feature] > threshold]
            left_coalition = set([players[i] for i in left_indices])
            right_coalition = set([players[i] for i in right_indices])
            left_shapely = shapely_value(matrix, left_coalition)
            right_shapely = shapely_value(matrix, right_coalition)
            total_shapely = left_shapely + right_shapely

            if total_shapely > max_shapely:
                best_feature = feature
                best_threshold = threshold
                max_shapely = total_shapely

    left_indices = [i for i in range(len(X)) if X[i][best_feature] <= best_threshold]
    right_indices = [i for i in range(len(X)) if X[i][best_feature] > best_threshold]
    left_X, left_y = X[left_indices], y[left_indices]
    right_X, right_y = X[right_indices], y[right_indices]
    left_players = [players[i] for i in left_indices]
    right_players = [players[i] for i in right_indices]

    return {'feature': best_feature, 'threshold': best_threshold,
            'left': decision_tree(left_X, left_y, matrix, left_players),
            'right': decision_tree(right_X, right_y, matrix, right_players)}

# Define the custom splitting function that uses Shapley value
def shapley_split(X, y, features, random_state=None):
    # Calculate the Shapley value for each feature
    shapley_values = [shapely_value(X, y, i) for i in range(X.shape[1])]

    # Choose the feature with the highest Shapley value as the splitting feature
    best_feature_idx = np.argmax(shapley_values)
    best_feature = features[best_feature_idx]

    # Return the chosen feature as the splitting criterion
    return best_feature

# Define the decision tree with Shapley value as the splitting criterion
dt = DecisionTreeClassifier(criterion=lambda X, y, features: shapley_split(X, y, features), random_state=42)

# Define the random forest with three decision trees
rf = RandomForestClassifier(n_estimators=3, criterion=lambda X, y, features: shapley_split(X, y, features), random_state=42)

# Fit the random forest to the training data
print("x",X_train)
print("y",y_train)
rf.fit(X_train, y_train)

# Evaluate the accuracy of the random forest on the testing data
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Calculate the feature importances using permutation importance
importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
for i, imp in enumerate(importance.importances_mean):
    print("Feature {}: {:.2f}%".format(i, imp * 100))

