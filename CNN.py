import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras import layers, models

# Assuming your data is in a directory with images and annotations folders
data_dir = 'NEU-DET/train'
images_dir = os.path.join(data_dir, 'images')

# Set image dimensions
img_width, img_height = 128, 128

# Initialize empty lists to store data and labels
data = []
labels = []

# Load and preprocess the data
label_encoder = LabelEncoder()

# Iterate through defect types
for defect_type in os.listdir(images_dir):
    defect_type_folder = os.path.join(images_dir, defect_type)
    
    # Check if the item in the directory is a subdirectory
    if os.path.isdir(defect_type_folder):
        
        # Iterate through images in each defect type
        for image_name in os.listdir(defect_type_folder):
            
            # Check if the file is a JPEG image
            if image_name.endswith('.jpg'):
                
                # Construct the full path to the image
                image_path = os.path.join(defect_type_folder, image_name)
                
                # Load and preprocess the image
                image = Image.open(image_path).resize((img_width, img_height))  # Load and resize image
                image_array = np.array(image) / 255.0  # Convert to NumPy array and normalize to [0, 1]
                
                # Append the preprocessed image array to the data list
                data.append(image_array)
                
                # Append the corresponding defect type (label) to the labels list
                labels.append(defect_type)


# Convert lists to numpy arrays
X = np.array(data)
y = np.array(labels)

# Encode the labels
y_encoded = label_encoder.fit_transform(y)
print(y_encoded)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# Build the CNN model
model = models.Sequential()  # Create a sequential model (linear stack of layers)

# Add the first convolutional layer with 32 filters, each of size (3, 3), ReLU activation, and input shape
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))

# Add max pooling layer with pool size (2, 2)
model.add(layers.MaxPooling2D((2, 2)))

# Add the second convolutional layer with 64 filters, each of size (3, 3), and ReLU activation
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Add another max pooling layer with pool size (2, 2)
model.add(layers.MaxPooling2D((2, 2)))

# Add the third convolutional layer with 128 filters, each of size (3, 3), and ReLU activation
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# Add a third max pooling layer with pool size (2, 2)
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output to a 1D array before feeding into dense layers
model.add(layers.Flatten())

# Add a dense layer with 128 units and ReLU activation
model.add(layers.Dense(128, activation='relu'))

# Add the output layer with units equal to the number of classes and softmax activation
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f'Test accuracy: {test_acc}')
print(f'Test Loss: {test_loss}')
