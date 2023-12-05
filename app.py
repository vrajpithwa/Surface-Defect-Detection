from flask import Flask, render_template, request
import joblib
from knntesting import predict_defect_type
from NBtesting import predict_defect_typeNB
from RFtesting import predict_defect_typeRF
from CNNtesting import load_trained_model, predict_defect


import os
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input file from the form
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # Save the uploaded file
            file_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(file_path)

            # Make a prediction using the model
            predicted_defect_type = predict_defect_type(file_path)
       
            # Pass the result to the template
            return render_template('result.html', result=predicted_defect_type)

    # Handle the case where no file was uploaded or an error occurred
    return render_template('error.html', message='Error in prediction')

@app.route('/predict_naive_bayes', methods=['POST'])
def predict_naive_bayes_route():
    if request.method == 'POST':
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            file_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(file_path)

            # Make a prediction using the Naive Bayes model
            predicted_defect_type = predict_defect_typeNB(file_path)

            return render_template('result.html', result=predicted_defect_type)

    return render_template('error.html', message='Error in prediction')



@app.route('/predict_random_forest', methods=['POST'])
def predict_random_forest_route():
    if request.method == 'POST':
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            file_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(file_path)

            # Make a prediction using the RandomForest model
            predicted_defect_type = predict_defect_typeRF(file_path)

            return render_template('result.html', result=predicted_defect_type)

    return render_template('error.html', message='Error in prediction')



@app.route('/predict_cnn', methods=['POST'])
def predict_cnn_route():
    if request.method == 'POST':
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            file_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(file_path)

            # Load the trained CNN model
            trained_model = load_trained_model('CNN_SDD.h5')  # Replace with the actual path

            # Load the label encoder with its classes
            label_encoder = joblib.load('CNN_label_encoder.pkl')  # Replace with the actual path

            # Make a prediction using the CNN model
            predicted_defect_type = predict_defect(file_path, trained_model, label_encoder)

            return render_template('result.html', result=predicted_defect_type)

    return render_template('error.html', message='Error in prediction')

if __name__ == '__main__':
    app.run(debug=True)
