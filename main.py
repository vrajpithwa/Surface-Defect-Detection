import os
import json
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Assuming your data is in a directory with images and annotations folders
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
                # Load image
                image = Image.open(image_path)
                image_array = np.array(image).flatten()
                data.append(image_array)
                labels.append(defect_type)

# Convert lists to numpy arrays
X = np.array(data)
y = np.array(labels)

# Ensure that there are enough samples for training and testing
if len(set(y)) > 1:
    # Encode the labels
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    rf_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Print additional metrics such as classification report
    print('Classification Report:')
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
else:
    print("Not enough samples to split the dataset.")
