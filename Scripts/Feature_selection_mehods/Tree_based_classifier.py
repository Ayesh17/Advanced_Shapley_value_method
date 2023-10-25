import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols):

    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    # Create a decision tree classifier with a fixed random seed of 42
    tree_clf = DecisionTreeClassifier(random_state=42)

    # Train the decision tree classifier on the training set X_train
    tree_clf.fit(X_train, y_train)

    # Calculate the feature importances of the trained decision tree classifier
    importances = tree_clf.feature_importances_

    # Sort the features in descending order of importance
    indices = np.argsort(importances)[::-1]

    # Determine the number of features to keep by selecting the top 70% of the most important features
    num_features = int(np.ceil(0.7 * len(X_train[0])))

    # Select the top num_features features to keep
    top_indices = indices[:num_features]

    # Create new feature sets X_train_reduced and X_test_reduced by selecting the top features
    X_train_reduced = X_train[:, top_indices]
    X_test_reduced = X_test[:, top_indices]

    return X_train_reduced, X_test_reduced
