# new_predictor.py

import os
import joblib
import numpy as np
from PIL import Image
label_encoder = joblib.load('label_encoder.joblib')
pipeline = joblib.load('NB_model.joblib')


def predict_defect_typeNB(new_image_path):
    # Load and preprocess the new input data
    new_image_array = np.array(Image.open(new_image_path)).flatten()    

    # Reshape the input array to match the shape expected by the model
    new_image_array = new_image_array.reshape(1, -1)

    # Make predictions using the trained model
    predicted_label_encoded = pipeline.predict(new_image_array)

    # Decode the predicted label
    predicted_label = label_encoder.inverse_transform(predicted_label_encoded)

    return predicted_label[0]
    
# if __name__ == "__main__":
#     # Example usage
#     new_image_path = 'crazing_242.jpg'
#     predicted_defect_type = predict_defect_typeNB(new_image_path)
#     print(f'The predicted defect type is: {predicted_defect_type}')
    