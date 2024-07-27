from flask import Flask, render_template, request, redirect, url_for , jsonify
import pickle
import os
import pickle
import numpy as np
import fitz  # PyMuPDF
import os
import joblib
app = Flask(__name__)

# Load models
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

breast_cancer_model = pickle.load(open(f'{working_dir}/saved_models/breast_cancer_model.sav', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/breast_cancer', methods=['GET', 'POST'])
def breast_cancer():
    diagnosis = ''
    if request.method == 'POST':
        try:
            user_input = [
                float(request.form['radius_mean']),
                float(request.form['texture_mean']),
                float(request.form['perimeter_mean']),
                float(request.form['area_mean']),
                float(request.form['smoothness_mean']),
                float(request.form['compactness_mean']),
                float(request.form['concavity_mean']),
                float(request.form['concave_points_mean']),
                float(request.form['symmetry_mean']),
                float(request.form['fractal_dimension_mean']),
                float(request.form['radius_se']),
                float(request.form['texture_se']),  # Corrected input name
                float(request.form['perimeter_se']),  # Corrected input name
                float(request.form['area_se']),
                float(request.form['smoothness_se']),
                float(request.form['compactness_se']),
                float(request.form['concavity_se']),
                float(request.form['concave_points_se']),
                float(request.form['symmetry_se']),
                float(request.form['fractal_dimension_se']),
                float(request.form['radius_worst']),
                float(request.form['texture_worst']),
                float(request.form['perimeter_worst']),
                float(request.form['area_worst']),
                float(request.form['smoothness_worst']),
                float(request.form['compactness_worst']),
                float(request.form['concavity_worst']),
                float(request.form['concave_points_worst']),
                float(request.form['symmetry_worst']),
                float(request.form['fractal_dimension_worst'])
            ]
            prediction = breast_cancer_model.predict([user_input])
            if prediction[0] == 1:
                diagnosis = 'Malignant'
            else:
                diagnosis = 'Benign'
        except ValueError:
            diagnosis = 'Invalid input. Please enter numerical values.'

    return render_template('breast_cancer.html', diagnosis=diagnosis)


@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    diagnosis = ''
    if request.method == 'POST':
        try:
            user_input = [
                float(request.form['Pregnancies']),
                float(request.form['Glucose']),
                float(request.form['BloodPressure']),
                float(request.form['SkinThickness']),
                float(request.form['Insulin']),
                float(request.form['BMI']),
                float(request.form['DiabetesPedigreeFunction']),
                float(request.form['Age'])
            ]
            prediction = diabetes_model.predict([user_input])
            if prediction[0] == 1:
                diagnosis = 'The person is diabetic'
            else:
                diagnosis = 'The person is not diabetic'
        except ValueError:
            diagnosis = 'Invalid input. Please enter numerical values.'

    return render_template('diabetes.html', diagnosis=diagnosis)

@app.route('/heart_disease', methods=['GET', 'POST'])
def heart_disease():
    diagnosis = ''
    if request.method == 'POST':
        try:
            user_input = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['cp']),
                float(request.form['trestbps']),
                float(request.form['chol']),
                float(request.form['fbs']),
                float(request.form['restecg']),
                float(request.form['thalach']),
                float(request.form['exang']),
                float(request.form['oldpeak']),
                float(request.form['slope']),
                float(request.form['ca']),
                float(request.form['thal'])
            ]
            prediction = heart_disease_model.predict([user_input])
            if prediction[0] == 1:
                diagnosis = 'The person is having heart disease'
            else:
                diagnosis = 'The person does not have any heart disease'
        except ValueError:
            diagnosis = 'Invalid input. Please enter numerical values.'

    return render_template('heart_disease.html', diagnosis=diagnosis)



if __name__ == '__main__':
    app.run(debug=True)
