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
data_dir = 'Bridge_Crack_Image/DBCC_Training_Data_Set/train'
images_dir = os.path.join(data_dir)

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


# confusion matrix to see how well the model is performing on each defect type.
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()





# Assuming you have a new image to predict
new_image_path = 'NEU-DET/train/images/patches/patches_4.jpg'
new_image = Image.open(new_image_path)
new_image_array = np.array(new_image).flatten()

# Standardize the features (using the same scaler from the training set)
new_image_array_standardized = scaler.transform(new_image_array.reshape(1, -1))

# Make predictions
new_prediction = rf_classifier.predict(new_image_array_standardized)
print(f'Predicted defect type: {label_encoder.inverse_transform(new_prediction)}')
