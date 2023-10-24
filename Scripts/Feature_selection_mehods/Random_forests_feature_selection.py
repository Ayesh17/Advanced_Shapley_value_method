import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Define the Random Forest model
rf = RandomForestClassifier(random_state=42)

def get_best_features(X_train, X_test, y_train, y_test):
    # Fit the Random Forest model to the training data
    rf.fit(X_train, y_train)

    # Get feature importances from the Random Forest model
    importances = rf.feature_importances_

    # Sort the features by importance
    indices = np.argsort(importances)[::-1]

    # Determine the number of features to keep
    num_features = int(np.ceil(0.7 * len(X_train[0])))

    # Select the top features to keep
    top_indices = indices[:num_features]
    X_train_reduced = X_train[:, top_indices]
    X_test_reduced = X_test[:, top_indices]

    return X_train_reduced, X_test_reduced
