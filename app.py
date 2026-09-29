from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained ML model
model = joblib.load("models/diabetes_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from the HTML form
        pregnancies = float(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        blood_pressure = float(request.form["blood_pressure"])
        skin_thickness = float(request.form["skin_thickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        diabetes_pedigree = float(request.form["diabetes_pedigree"])
        age = float(request.form["age"])

        # Backend validation
        if not 0 <= pregnancies <= 20:
            raise ValueError("Pregnancies must be between 0 and 20.")

        if not 1 <= glucose <= 300:
            raise ValueError("Glucose must be between 1 and 300.")

        if not 1 <= blood_pressure <= 200:
            raise ValueError("Blood pressure must be between 1 and 200.")

        if not 0 <= skin_thickness <= 100:
            raise ValueError("Skin thickness must be between 0 and 100.")

        if not 0 <= insulin <= 900:
            raise ValueError("Insulin must be between 0 and 900.")

        if not 1 <= bmi <= 70:
            raise ValueError("BMI must be between 1 and 70.")

        if not 0 <= diabetes_pedigree <= 3:
            raise ValueError(
                "Diabetes Pedigree Function must be between 0 and 3."
            )

        if not 1 <= age <= 120:
            raise ValueError("Age must be between 1 and 120.")

        # Create DataFrame using the same column names
        # used during model training
        input_data = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree,
            "Age": age
        }])

        # Make prediction
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "Diabetes Prediction: Positive"
        else:
            result = "Diabetes Prediction: Negative"

        return render_template(
            "index.html",
            prediction=result
        )

    except ValueError as error:

        return render_template(
            "index.html",
            error=str(error)
        )


if __name__ == "__main__":
    app.run(debug=True)