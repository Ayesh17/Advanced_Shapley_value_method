from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix

def calc_accuracy(X_train_reduced, X_test_reduced, y_train, y_test):
    # Define the decision tree classifier with a fixed random seed of 42
    clf = DecisionTreeClassifier(random_state=42)

    # Fit the decision tree classifier to the reduced training data
    clf.fit(X_train_reduced, y_train)

    # Evaluate the accuracy of the decision tree classifier on the reduced testing data
    accuracy = clf.score(X_test_reduced, y_test)

    # Predict the classes of the reduced testing data
    y_pred = clf.predict(X_test_reduced)

    # Calculate the F1 score of the logistic regression classifier on the reduced testing data
    f1 = f1_score(y_test, y_pred)

    # Calculate the confusion matrix of the predicted classes
    cm = confusion_matrix(y_test, y_pred)

    # Calculate sensitivity and specificity from the confusion matrix
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return accuracy, precision, sensitivity, f1, specificity
