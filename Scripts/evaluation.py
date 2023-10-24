import os

from Scripts.ML_models import Decision_tree as dec_tree, Gradient_boosting as gr_boost, Logistic_Regression as log_reg, \
    Original_RF as rf, Neural_Network as neur


def evaluation(model, X_train_reduced, X_test_reduced, y_train, y_test):
    # Generate logistic regression with reduced features
    print()
    print("Logistic Regression")
    acc_lr, pre_lr, rec_lr, f1_lr, spe_lr = log_reg.calc_accuracy(X_train_reduced, X_test_reduced, y_train, y_test)

    #
    # Generate decision tree with reduced features
    # print()
    # print("Decision Tree")
    acc_dt, pre_dt, se_dt, f1_dt, spe_dt = dec_tree.calc_accuracy(X_train_reduced, X_test_reduced, y_train, y_test)


    # Generate the random forest with reduced features
    # print()
    # print("Random Forest")
    no_trees = 50
    acc_rf, pre_rf, rec_rf, f1_rf, spe_rf = rf.calc_accuracy(X_train_reduced, X_test_reduced, y_train, y_test, no_trees)


    # Generate the gradient boosting with reduced features
    # print()
    # print("Gradient Boosting")
    no_trees = 100
    acc_gb, pre_gb, rec_gb, f1_gb, spe_gb = gr_boost.calc_accuracy(X_train_reduced, X_test_reduced, y_train, y_test, no_trees)


    # Generate the neural network with reduced features
    # print()
    # print("Neural Network")
    no_epochs = 100
    acc_ne, pre_ne, rec_ne, f1_ne, spe_ne = neur.calc_accuracy(X_train_reduced, X_test_reduced, y_train, y_test, no_epochs)


    # header = "Evaluation\n\n"
    header = "Evaluation\n\n"
    data_rows = [
        '%s\n\n' %(model),
        '%s, %s, %s, %s, %s, %s \n' % ("Model", "Accuracy", "Precision", "Recall", "F1-score", "Specificity"),
        'Logistic_regression, %f, %f, %f, %f, %f\n' % (acc_lr, pre_lr, rec_lr, f1_lr, spe_lr),
        'Decision Tree, %f, %f, %f, %f, %f\n' % (acc_dt, pre_dt, se_dt, f1_dt, spe_dt),
        'Random Forest, %f, %f, %f, %f, %f\n' % (acc_rf, pre_rf, rec_rf, f1_rf, spe_rf),
        'Gradient Boosting, %f, %f, %f, %f, %f\n' % (acc_gb, pre_gb, rec_gb, f1_gb, spe_gb),
        'Neural Network, %f, %f, %f, %f, %f\n\n\n' % (acc_ne, pre_ne, rec_ne, f1_ne, spe_ne)
    ]

    filename = '../evaluation_results.csv'

    # Check if the file already exists
    if os.path.exists(filename):
        # Append new data as new columns
        with open(filename, 'a') as openfile:
            for row in data_rows:
                openfile.write(row)
    else:
        # Create a new file and write data with header
        with open(filename, 'w') as openfile:
            openfile.write(header)
            for row in data_rows:
                openfile.write(row)

# Example usage:
# evaluation(X_train_reduced, X_test_reduced, y_train, y_test)
