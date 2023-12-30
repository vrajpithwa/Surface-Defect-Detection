

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from keras import layers, models

# Set the seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the data directory
data_dir = 'NEU-DET/train'
images_dir = os.path.join(data_dir, 'images')

# Initialize empty lists to store data and labels
data = []
labels = []

# Load and preprocess the data
label_encoder = LabelEncoder()
for defect_type in os.listdir(images_dir):
    defect_type_folder = os.path.join(images_dir, defect_type)
    if os.path.isdir(defect_type_folder):
        for image_name in os.listdir(defect_type_folder):
            if image_name.endswith('.jpg'):  # Check if the file is an image
                image_path = os.path.join(defect_type_folder, image_name)
                # Load image, resize, and normalize
                image = Image.open(image_path).resize((224, 224))  # Adjust the size as needed
                image_array = np.array(image) / 255.0  # Normalize pixel values to the range [0, 1]
                data.append(image_array)
                labels.append(defect_type)

# Convert lists to numpy arrays
X = np.array(data)
y = np.array(labels)

# Encode the labels
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Create a simple CNN model
model = models.Sequential()#stacking layers in seq  (convo, max pooling , convo, fully conn)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3))) 
 # 32 = no. of filter/op chann, 3,3 is siz of filter
model.add(layers.MaxPooling2D((2, 2))) # 2,2 filter size 
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu')) #128 no. of neurons per lay
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))


# Compile the model with legacy Adam optimizer
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the performance on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)


# Make predictions on the test set
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Decode predictions and true labels for printing the classification report
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

# Print additional metrics such as classification report
print('Classification Report:')
print(classification_report(y_test_labels, y_pred_labels, target_names=label_encoder.classes_))
print(f'Test Accuracy: {test_accuracy}')
model.save('CNN_SDD.h5')

# Save the label encoder
import joblib
joblib.dump(label_encoder, 'CNN_label_encoder.pkl')
