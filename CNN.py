import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

# Assuming your data is in a directory with images and annotations folders
data_dir = 'NEU-DET/validation'
images_dir = os.path.join(data_dir, 'images')

# Set image dimensions
img_width, img_height = 128, 128

# Initialize empty lists to store data and labels
data = []
labels = []

# Load and preprocess the data
label_encoder = LabelEncoder()
for defect_type in os.listdir(images_dir):
    defect_type_folder = os.path.join(images_dir, defect_type)
    if os.path.isdir(defect_type_folder):
        for image_name in os.listdir(defect_type_folder):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(defect_type_folder, image_name)
                # Load and preprocess image
                image = Image.open(image_path).resize((img_width, img_height))
                image_array = np.array(image) / 255.0  # Normalize to [0, 1]
                data.append(image_array)
                labels.append(defect_type)

# Convert lists to numpy arrays
X = np.array(data)
y = np.array(labels)

# Encode the labels
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
