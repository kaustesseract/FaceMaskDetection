# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 18:32:05 2021

@author: kaust
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


learning_rate = 1e-4
epochs = 50
batch_size = 32

directory = r"C:\Users\kaust\Desktop\Deep_learning\Projects\FaceMaskDetection-main\dataset"
category = ["with_mask", "without_mask"]

data = []
labels = []

for categories in category:
    path = os.path.join(directory,categories)
    for img in os.listdir(path):
        path_img = os.path.join(path, img)
        image = load_img(path_img, target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(categories)

# converting into one-hot encoding

label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.30, stratify=labels, random_state=42)

aug = ImageDataGenerator(
        rotation_range = 25,
        zoom_range = 0.12,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

mob_v2 = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))) # importing the mobile v2 pre-trained model

# defining the top layer
last_layer = mob_v2.output
last_layer = AveragePooling2D(pool_size=(7,7))(last_layer)
last_layer = Flatten(name="flatten")(last_layer)
last_layer = Dense(128, activation="relu")(last_layer)
last_layer = Dropout(0.5)(last_layer)
last_layer = Dense(2, activation="softmax")(last_layer)

model = Model(inputs=mob_v2.input, outputs=last_layer) # creating the model

for layers in mob_v2.layers: # freezing the other layers
    layers.trainable = False

opt = Adam(lr=learning_rate, decay=learning_rate/epochs)
model.compile(loss="binary_crossentropy", optimizer = opt, metrices=["accuracy"])

# training the network

train_model = model.fit(aug.flow(trainX, trainY, batch_size=batch_size,), steps_per_epoch=len(trainX) // batch_size, 
                        validation_data=(testX, testY), validation_steps=len(testX) // batch_size, epochs=epochs)

prediction = model.predict(testX, batch_size=batch_size)

max_prediction = np.argmax(prediction, axis=1)

print(classification_report(testY.argmax(axis=1), max_prediction,
	target_names=label_binarizer.classes_))

model.save("mask_detector.model", save_format="h5")

