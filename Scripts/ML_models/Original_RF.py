#imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

def calc_accuracy(X_train_reduced,X_test_reduced, y_train, y_test, no_trees):
    # Define the random forest with decision trees
    rf = RandomForestClassifier(n_estimators=no_trees, random_state=7)

    # Fit the random forest to the reduced training data
    rf.fit(X_train_reduced, y_train)

    # Evaluate the accuracy of the random forest on the reduced testing data
    y_pred = rf.predict(X_test_reduced)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return accuracy, precision, sensitivity, f1, specificity
