import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Scripts.Feature_selection_mehods import recursive_feature_elimination as rfe, \
    Random_forests_feature_selection as rffs, Univariate_feature_selction as univariate, \
    Tree_based_classifier as tree_cls, Shapely_value as shap, Adv_Shapley_value as sh_b
from Scripts import evaluation as evl

# Load the data from the CSV file
# df = pd.read_csv('cat_dog_data.csv')
# df = pd.read_csv('haberman.csv')
# df = pd.read_csv('haberman_edited.csv')
# df = pd.read_csv('diabetes.csv')
# df = pd.read_csv('microgrid_data.csv')3 layers.
# df = pd.read_csv('glass.csv')
# df = pd.read_csv('heart_diseases_dataset.csv')
# df = pd.read_csv('wine_quality_dataset.csv')pima


# This works
df = pd.read_csv('../Dataset/diabetes.csv')

# df = pd.read_csv('../Dataset/titanic_1.csv')
# df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
# df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
# # Drop rows with missing values
# df = df.dropna()


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

# print("X_train",X_train)

#Get the preprocessed dataset to select the best features

X_train_reduced, X_test_reduced = X_train, X_test


# Delete the evaluuation_results csv file if exists
filename = '../Results/evaluation_results.csv'
if os.path.exists(filename):
    os.remove(filename)


# #Univariate feature selection
# model = "Univariate"
# X_train_reduced, X_test_reduced = univariate.get_best_features(X_train, X_test, y_train, y_test)
# evl.evaluation(model, X_train_reduced, X_test_reduced, y_train, y_test)
#
# # tree classifier
# model = "Tree Classifier"
# X_train_reduced, X_test_reduced = tree_cls.get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols)
# evl.evaluation(model, X_train_reduced, X_test_reduced, y_train, y_test)
#
# #RFE
# model = "RFE"
# X_train_reduced, X_test_reduced = rfe.get_best_features(X_train, X_test, y_train, y_test)
# evl.evaluation(model, X_train_reduced, X_test_reduced, y_train, y_test)
#
# #RFFS
# model = "RFFS"
# X_train_reduced, X_test_reduced = rffs.get_best_features(X_train, X_test, y_train, y_test)
# evl.evaluation(model, X_train_reduced, X_test_reduced, y_train, y_test)
#
# #Shapley value method - this is the shapley method I guess
# model = "Original"
# evl.evaluation(model, X_train, X_test, y_train, y_test)
#
#Shapley value method - this is the shapley method I guess
model = "Shapley"
X_train_reduced, X_test_reduced = shap.get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols)
evl.evaluation(model, X_train_reduced, X_test_reduced, y_train, y_test)

#Shapley value method by bard  -  this is the method we propose in the paper
model = "Adv. Shapley"
X_train_reduced, X_test_reduced = sh_b.get_best_features(X, y, X_train, X_test, y_train, y_test, num_cols)
evl.evaluation(model, X_train_reduced, X_test_reduced, y_train, y_test)



