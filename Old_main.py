import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Scripts.Feature_selection_mehods import Shap_bard as sh_b

from Scripts.ML_models import Decision_tree as dec_tree, Gradient_boosting as gr_boost, Logistic_Regression as log_reg, \
    Original_RF as rf, Neural_Network as neur

# Load the data from the CSV file
# df = pd.read_csv('cat_dog_data.csv')
# df = pd.read_csv('haberman.csv')
# df = pd.read_csv('haberman_edited.csv')
# df = pd.read_csv('diabetes.csv')
# df = pd.read_csv('microgrid_data.csv')3 layers.
# df = pd.read_csv('glass.csv')
# df = pd.read_csv('heart_diseases_dataset.csv')
# df = pd.read_csv('wine_quality_dataset.csv')

# df = pd.read_csv('wisconsin_dataset.csv')
# df['diagnosis'].replace(['B', 'M'], [0, 1], inplace=True)

# print(df)

# df = pd.read_csv('iris_dataset.csv')
# df['species'].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2], inplace=True)

# This works
df = pd.read_csv('titanic_1.csv')
df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
# Drop rows with missing values
df = df.dropna()

# df = pd.read_csv('adult.csv')
# # df = df.drop(columns=['fnlwgt', 'education'])  # drop unnecessary columns
# df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)  # convert income to binary label
# df = pd.get_dummies(df)  # one-hot encode categorical variables
# print(df.shape)
# X = df.drop(columns=['income'])
# y = df['income']

# df = pd.read_csv('bank.csv')
# df['deposit'] = df['deposit'].apply(lambda x: 1 if x.strip() == 'yes' else 0)  # convert income to binary label
# df = pd.get_dummies(df)  # one-hot encode categorical variables
# print(df.shape)
# X = df.drop(columns=['deposit'])
# y = df['deposit']

# Get the x and y values as NumPy arrays
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


num_cols = df.shape[1]

# Ensure X is a 2D numpy array and y is a 1D numpy array
X = np.atleast_2d(X)
y = np.squeeze(y)

# Print the shape of x and y to verify they are correct
print('x shape:', X.shape)
print('y shape:', y.shape)

# Load your data and split it into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train",X_train)

#Get the preprocessed dataset to select the best features

X_train_reduced, X_test_reduced = X_train, X_test

# #Shapley value method - this is the shapley method I guess
# X_train_reduced, X_test_reduced = shap.get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols)

# #Shapley value method
# X_train_reduced, X_test_reduced = shap_2.get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols)

#Shapley value method by bard  -  this is the method we propose in the paper
X_train_reduced, X_test_reduced = sh_b.get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols)

# #Nucleus value method
# X_train_reduced, X_test_reduced = nucleus.get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols)


# #Nucleus value method 2
# X_train_reduced, X_test_reduced = nuc2.get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols)


# #Shapley and nucleus mixed method
# X_train_reduced, X_test_reduced = shap_nuc.get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols)


# #Nucleus value method 2 -> Nuc Bard
# X_train_reduced, X_test_reduced = nuc_bard.get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols)

# #Nucleus value 4
# X_train_reduced, X_test_reduced = nuc_4.get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols)

# #Univariate feature selection
# X_train_reduced, X_test_reduced = univariate.get_best_features(X_train, X_test, y_train, y_test)

# # tree classifier
# X_train_reduced, X_test_reduced = tree_cls.get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols)

# #RFE
# X_train_reduced, X_test_reduced = rfe.get_best_features(X_train, X_test, y_train, y_test)

# #RFFS
# X_train_reduced, X_test_reduced = rffs.get_best_features(X_train, X_test, y_train, y_test)

print("X_train_reduced",X_train_reduced)
print("X_test_reduced",X_test_reduced)

#Generate logistic regression with reduced features
print()
print("Logistic Regression")
acc, precision, se, f1, sp = log_reg.calc_accuracy(X_train_reduced,X_test_reduced, y_train, y_test)
print("Accuracy: {:.2f}%".format(acc * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Sensitivity: {:.2f}%".format(se * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))
print("Specificity: {:.2f}%".format(sp * 100))

#Generate logistic regression with reduced features
print()
print("Decision Tree")
acc, precision, se, f1, sp = dec_tree.calc_accuracy(X_train_reduced,X_test_reduced, y_train, y_test)
print("Accuracy: {:.2f}%".format(acc * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Sensitivity: {:.2f}%".format(se * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))
print("Specificity: {:.2f}%".format(sp * 100))


#Generate the random forest with reduced features
print()
print("Random Forest")
no_trees =50
acc, precision, se, f1, sp = rf.calc_accuracy(X_train_reduced,X_test_reduced, y_train, y_test, no_trees)
print("Accuracy: {:.2f}%".format(acc * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Sensitivity: {:.2f}%".format(se * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))
print("Specificity: {:.2f}%".format(sp * 100))

#Generate the gradient boosting with reduced features
print()
print("gradient Boosting")
no_trees =100
acc, precision, se, f1, sp = gr_boost.calc_accuracy(X_train_reduced,X_test_reduced, y_train, y_test, no_trees)
print("Accuracy: {:.2f}%".format(acc * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Sensitivity: {:.2f}%".format(se * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))
print("Specificity: {:.2f}%".format(sp * 100))

#Generate the gradient boosting with reduced features
print()
print("Neural Network")
no_epochs = 100
acc, precision, se, f1, sp = neur.calc_accuracy(X_train_reduced,X_test_reduced, y_train, y_test, no_epochs)
print("Accuracy: {:.2f}%".format(acc * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Sensitivity: {:.2f}%".format(se * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))
print("Specificity: {:.2f}%".format(sp * 100))