import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set TensorFlow logging level to ERROR (2)

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import random
from tensorflow.random import set_seed
from keras.layers import Dense
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def smooth_curve(points, factor=0.7):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def calc_accuracy(X_train_reduced, X_test_reduced, y_train, y_test, num_epochs):
    # Set random seeds for reproducibility
    # np.random.seed(49)
    x = 42
    random.seed(x)
    set_seed(x)

    # Create the neural network model
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X_train_reduced.shape[1]))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train_reduced, y_train, epochs=num_epochs, batch_size=32, validation_data=(X_test_reduced, y_test), verbose=0)

    # Get the training accuracy values and smooth them
    training_acc = history.history['accuracy']
    smoothed_training_acc = smooth_curve(training_acc)

    # Get the validation accuracy values and smooth them
    validation_acc = history.history['val_accuracy']
    smoothed_validation_acc = smooth_curve(validation_acc)

    # Plot smoothed accuracy over epochs
    plt.plot(smoothed_training_acc, label='Training Accuracy')
    plt.plot(smoothed_validation_acc, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.show()

    # Evaluate the model on the testing data
    y_pred_prob = model.predict(X_test_reduced)
    y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate the sensitivity and specificity
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return accuracy, precision, sensitivity, f1, specificity
