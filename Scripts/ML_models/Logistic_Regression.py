from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def calc_accuracy(X_train_reduced, X_test_reduced, y_train, y_test):
    # Define the logistic regression classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)

    # Fit the logistic regression classifier to the reduced training data
    clf.fit(X_train_reduced, y_train)

    # Predict the classes of the reduced testing data
    y_pred = clf.predict(X_test_reduced)

    # Calculate the confusion matrix for the logistic regression classifier on the reduced testing data
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate the sensitivity and specificity of the logistic regression classifier on the reduced testing data
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Calculate the accuracy and F1 score of the logistic regression classifier on the reduced testing data
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, sensitivity, f1, specificity

