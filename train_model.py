import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# File paths
DATA_PATH = "data/diabetesdata.csv"
MODEL_PATH = "models/diabetes_model.pkl"


# Load dataset
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Columns where zero represents missing/invalid measurements
zero_invalid_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Columns where zero is valid
normal_columns = [
    "Pregnancies",
    "DiabetesPedigreeFunction",
    "Age"
]


# Preprocessing
preprocessor = ColumnTransformer([
    (
        "medical_imputer",
        SimpleImputer(
            missing_values=0,
            strategy="median"
        ),
        zero_invalid_columns
    ),
    (
        "normal_imputer",
        SimpleImputer(strategy="median"),
        normal_columns
    )
])


# Complete ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf"))
])


# Train model
model.fit(X_train, y_train)


# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)


# Save model
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")