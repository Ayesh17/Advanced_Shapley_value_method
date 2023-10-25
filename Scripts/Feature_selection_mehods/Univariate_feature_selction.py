import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def get_best_features(X_train, X_test, y_train, y_test):
    # Univariate feature selection
    selector = SelectPercentile(f_classif, percentile=70)
    X_train_reduced = selector.fit_transform(X_train, y_train)
    X_test_reduced = selector.transform(X_test)

    return X_train_reduced, X_test_reduced

