# Diabetes Prediction using Machine Learning

A machine learning web application that predicts the likelihood of diabetes based on selected patient health measurements.

The project uses a Support Vector Machine (SVM) with an RBF kernel and provides a Flask-based web interface for making predictions.

## Features

- Diabetes prediction using machine learning
- RBF Support Vector Machine classifier
- Data preprocessing and feature scaling
- Median imputation for missing/invalid measurements
- Responsive web interface
- Input validation
- Flask backend
- Saved trained model using Joblib
- Reproducible model training script

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Joblib
- HTML
- CSS
- Jupyter Notebook

## Dataset

The project uses a diabetes dataset containing 768 patient records and 8 input features.

### Input Features

1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age

### Target

`Outcome`

- `0` → No diabetes according to the dataset label
- `1` → Diabetes according to the dataset label

## Machine Learning Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Train/Test Split
   ↓
Invalid Zero-Value Handling
   ↓
Median Imputation
   ↓
Standard Scaling
   ↓
RBF SVM
   ↓
Model Evaluation
   ↓
Saved Model
   ↓
Flask Web Application