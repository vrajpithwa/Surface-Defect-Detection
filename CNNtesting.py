# defect_prediction.py

from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras import models
import joblib  # Import joblib for loading the label encoder

def load_trained_model(model_path):
    # Load the trained model from the specified path
    model = models.load_model(model_path)
    return model

def predict_defect(input_image_path, model, label_encoder):
    # Load and preprocess the input image
    input_image = Image.open(input_image_path).resize((224, 224))
    input_image_array = np.array(input_image) / 255.0
    input_image_array = np.expand_dims(input_image_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(input_image_array)
    predicted_class_index = np.argmax(predictions)

    # Decode the predicted class using the loaded label encoder
    predicted_class = label_encoder.classes_[predicted_class_index]

    return predicted_class

if __name__ == "__main__":
    # Specify the path to the trained model
    model_path = 'CNN_SDD.h5'  # Replace with the actual path

    # Specify the path to the input image
    input_image_path = 'pitted_surface_242.jpg'  # Replace with the actual path

    # Load the trained model
    trained_model = load_trained_model(model_path)

    # Load the label encoder with its classes
    label_encoder = joblib.load('CNN_label_encoder.pkl')  # Replace with the actual path

    # Make predictions
    predicted_defect = predict_defect(input_image_path, trained_model, label_encoder)

    print(f'Predicted Defect Class: {predicted_defect}')
