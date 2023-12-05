import os
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

# Set the seed for reproducibility
np.random.seed(42)

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
                # Load image and flatten it
                image_array = np.array(Image.open(image_path)).flatten()
                data.append(image_array)
                labels.append(defect_type)

# Convert lists to numpy arrays
X = np.array(data)
y = np.array(labels)

# Encode the labels
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a pipeline with scaling and classifier
pipeline = make_pipeline(StandardScaler(), rf_classifier)

# Train the classifier
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Print additional metrics such as classification report
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Perform cross-validation
cv_scores = cross_val_score(pipeline, X, y_encoded, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
print(f'Cross-Validation Mean Accuracy: {np.mean(cv_scores)}')



joblib.dump(pipeline, 'RandomForest_model.joblib')

