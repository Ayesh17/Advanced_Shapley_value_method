from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

def calc_accuracy(X_train_reduced, X_test_reduced, y_train, y_test, no_trees):
    # Define the gradient boosting classifier
    gbc = GradientBoostingClassifier(n_estimators=no_trees, random_state=42)

    # Fit the gradient boosting classifier to the reduced training data
    gbc.fit(X_train_reduced, y_train)

    # Evaluate the accuracy, F1 score, precision and recall of the gradient boosting classifier on the reduced testing data
    y_pred = gbc.predict(X_test_reduced)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Calculate the confusion matrix of the gradient boosting classifier on the reduced testing data
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate the sensitivity and specificity of the gradient boosting classifier on the reduced testing data
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return accuracy, precision, sensitivity, f1, specificity