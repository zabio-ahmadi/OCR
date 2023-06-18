import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.optimizers import Adam

data_folder = "static/data"

num_classes = 10

def train_model(model_name):

    """
    Trains a neural network model for digit recognition.

    Parameters:
        model_name (str): The name of the model file to save.

    Returns:
        None

    This function loads the labeled images for each digit from  assets/data. 
    
    Note: This function assumes that the data folder contains subfolders for each digit (0 to 9), and each subfolder contains PNG images of the corresponding digit.
    """
        

    images = []
    labels = []

    # Iterate over each digit from 0 to 9
    for digit in range(num_classes):
        digit_folder = os.path.join(data_folder, str(digit))
        for filename in os.listdir(digit_folder):
            if filename.endswith(".png"):
                img = cv2.imread(
                    os.path.join(digit_folder, filename), cv2.IMREAD_GRAYSCALE
                )
                img = cv2.resize(img, (20, 20))
                images.append(img)
                labels.append(digit)

    images = np.array(images)
    labels = np.array(labels)

    images = images / 255.0  # Normalize pixel values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    # # Reshape the data to match the expected input shape of the model [samples][width][height][channels]
    X_train = X_train.reshape((X_train.shape[0], 20, 20, 1)).astype("float32")
    X_test = X_test.reshape((X_test.shape[0], 20, 20, 1)).astype("float32")

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # sequenctial model 
    model = Sequential()

    # Add layers to the model 
    # convertir les images 2D en un vecteur 1D
    model.add(Flatten(input_shape=(20, 20)))
    # les couches cachées
    model.add(Dense(784, activation="relu"))
    model.add(Dense(400, activation="relu"))
    # Output layer
    model.add(Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"]
    )

    # Train the model on the training data
    model.fit(
        X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32
    )

    model.save(model_name, overwrite=True)
