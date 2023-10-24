import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def get_best_features(X_train, X_test, y_train, y_test):
    # Define the logistic regression model
    logreg = LogisticRegression()

    # Define the RFE selector to select the top 70% of features
    selector = RFE(logreg, n_features_to_select=int(np.ceil(0.7 * len(X_train[0]))))

    # Fit the RFE selector to the training data
    selector.fit(X_train, y_train)

    # Apply the RFE selector to the training and testing data
    X_train_reduced = selector.transform(X_train)
    X_test_reduced = selector.transform(X_test)

    return X_train_reduced, X_test_reduced
