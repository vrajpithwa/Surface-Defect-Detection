from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from keras import models
import joblib
from CNNtesting import load_trained_model, predict_defect

app = Flask(__name__)

# Load the trained model and label encoder
model_path = 'CNN_SDD.h5'  # Replace with the actual path
trained_model = load_trained_model(model_path)

label_encoder = joblib.load('CNN_label_encoder.pkl')  # Replace with the actual path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file from the form
        uploaded_file = request.files['file']

        # Save the file to a temporary location
        temp_path = 'temp.jpg'
        uploaded_file.save(temp_path)

        # Make predictions using your model function
        predicted_defect = predict_defect(temp_path, trained_model, label_encoder)

        # Pass the result to the template
        return render_template('result.html', predicted_defect=predicted_defect)

if __name__ == '__main__':
    app.run(debug=True)
