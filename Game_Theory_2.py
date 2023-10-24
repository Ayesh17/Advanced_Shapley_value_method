import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19, ResNet50
from tensorflow.keras.layers import Input, Dense, concatenate, Reshape
from tensorflow.keras.models import Model

def cooperative_game(weights):
    # Reshape weights to a 3D array
    num_features = 27
    height = 16
    width = 4
    w = np.reshape(weights, (height, width, num_features))

    game = 0
    for i in range(height):
        for j in range(width):
            w2 = w.copy()
            w2[i, j, :] = 0
            game += np.sum(w * w2)
    return game


# Load the datasets for training and testing
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Resize the images to (224, 224)
train_images = tf.image.resize(train_images, (224, 224)).numpy()
test_images = tf.image.resize(test_images, (224, 224)).numpy()

# Add a batch dimension to the input data
train_images = np.expand_dims(train_images, axis=0)
test_images = np.expand_dims(test_images, axis=0)

# Add a batch dimension to the input data
train_images = np.expand_dims(train_images, axis=0)
test_images = np.expand_dims(test_images, axis=0)

# Add a batch dimension to the input data
train_images = train_images.reshape((50000, 224, 224, 3))
test_images = test_images.reshape((10000, 224, 224, 3))


# One-hot encode the labels
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=100)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=100)

# Initialize the CNN models
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg19_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Define the utility attribute for each model
vgg16_model.utility = 0.0
vgg19_model.utility = 0.0
resnet50_model.utility = 0.0

# Create a list to hold the models
models = [vgg16_model, vgg19_model, resnet50_model]



# Freeze the weights of the CNN models to avoid overfitting
for model in models:
    for layer in model.layers:
        layer.trainable = False
        layer.utility = 0.0

# Define the input layer
input_layer = Input(shape=(224, 224, 3))

# Get the output from each CNN model
outputs = [model(input_layer) for model in models]

# Concatenate the outputs
merged_output = concatenate(outputs, axis=-1)

# Reshape the output
merged_output = Reshape((1, 1, -1))(merged_output)

# Define the output layer
output_layer = Dense(100, activation='softmax')(merged_output)

# Define the model
model = Model(inputs=input_layer, outputs=output_layer)

# Train the model using cooperative game theory
batch_size = 16
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
num_batches = len(train_images) // batch_size
for epoch in range(100):
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        batch_images = train_images[batch_start:batch_end]
        batch_labels = train_labels[batch_start:batch_end]
        batch_labels = tf.reshape(batch_labels, (batch_size, 1, 1, -1))
        with tf.GradientTape() as tape:
            predictions = model(batch_images, training=True)
            loss = loss_fn(batch_labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Update the utilities of the CNN models using the cooperative game
        for model in models:
            for layer in model.layers:
                weights = layer.get_weights()
                if len(weights) > 0:
                    w = weights[0]
                    layer.utility = layer.utility + (cooperative_game(w) - layer.utility) / (epoch + 1)

        # Print the utility of each model at the end of each epoch
        for model in models:
            for layer in model.layers:
                print("Epoch {:03d}, Model {}, Layer {}, Utility {:.3f}".format(epoch + 1, model.name, layer.name, layer.utility))

    # Evaluate the model on the test data at the end of each epoch
    loss, accuracy = model.evaluate(test_images, test_labels)
    print("Epoch {:03d}, Test loss: {:.3f}, Test accuracy: {:.3f}".format(epoch + 1, loss, accuracy))

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_images, test_labels)
print("Test loss: {:.3f}, Test accuracy: {:.3f}".format(loss, accuracy))
