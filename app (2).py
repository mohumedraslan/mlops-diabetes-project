import streamlit as st
import pandas as pd
import joblib
import os

# Load trained model
model = joblib.load("diabetes_model_gbr.pkl")

st.title("Diabetes Progression Predictor")

st.markdown("""
This model predicts diabetes disease progression one year after baseline.
You can enter **real clinical values**, which are internally scaled to the model’s feature space.
""")

# -----------------------------
# Step 1: User inputs in real-world units
# -----------------------------
st.subheader("Patient Information (Real Values)")

age = st.number_input("Age (years)", min_value=20, max_value=80, value=50)
sex = st.selectbox("Sex", options=["Male", "Female"])  # We'll encode Male=1, Female=0
bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=50.0, value=25.0)
bp = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)

# S1–S6 are lab values; since the original dataset does not provide real units,
# we can still accept "relative units" to keep your app professional
s1 = st.number_input("S1 (rel. units)", min_value=-200.0, max_value=200.0, value=0.0)
s2 = st.number_input("S2 (rel. units)", min_value=-200.0, max_value=200.0, value=0.0)
s3 = st.number_input("S3 (rel. units)", min_value=-200.0, max_value=200.0, value=0.0)
s4 = st.number_input("S4 (rel. units)", min_value=-200.0, max_value=200.0, value=0.0)
s5 = st.number_input("S5 (rel. units)", min_value=-200.0, max_value=200.0, value=0.0)
s6 = st.number_input("S6 (rel. units)", min_value=-200.0, max_value=200.0, value=0.0)

# -----------------------------
# Step 2: Encode and scale inputs
# -----------------------------
# Sex encoding: Male=1, Female=0
sex_val = 1 if sex == "Male" else 0

# Feature means and stds (from original diabetes dataset)
# These numbers come from your df.describe() outputs
feature_means = {
    "age": 0.0, "sex": 0.0, "bmi": 0.0, "bp": 0.0,
    "s1": 0.0, "s2": 0.0, "s3": 0.0, "s4": 0.0, "s5": 0.0, "s6": 0.0
}

feature_stds = {
    "age": 0.0476, "sex": 0.0476, "bmi": 0.0476, "bp": 0.0476,
    "s1": 0.0476, "s2": 0.0476, "s3": 0.0476, "s4": 0.0476, "s5": 0.0476, "s6": 0.0476
}

# Scale user inputs
scaled_features = {
    "age": (age - 50.0) / 10.0,       # Replace 50 & 10 with actual dataset mean/std in years
    "sex": (sex_val - 0) / 1,          # simple encoding
    "bmi": (bmi - 25.0) / 5.0,         # Replace 25 & 5 with dataset mean/std
    "bp": (bp - 120.0) / 15.0,         # Replace with dataset mean/std
    "s1": s1 / 1.0,
    "s2": s2 / 1.0,
    "s3": s3 / 1.0,
    "s4": s4 / 1.0,
    "s5": s5 / 1.0,
    "s6": s6 / 1.0
}

# Convert to DataFrame for prediction
input_df = pd.DataFrame([[
    scaled_features["age"], scaled_features["sex"], scaled_features["bmi"], scaled_features["bp"],
    scaled_features["s1"], scaled_features["s2"], scaled_features["s3"], scaled_features["s4"],
    scaled_features["s5"], scaled_features["s6"]
]], columns=["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"])

# -----------------------------
# Step 3: Predict
# -----------------------------
prediction = None
if st.button("Predict Progression"):
    prediction = model.predict(input_df)[0]
    prediction = max(0, prediction)  # ensure non-negative
    st.success(f"Predicted Diabetes Progression Score: {prediction:.2f}")

# -----------------------------
# Step 4: Save prediction
# -----------------------------
if st.button("Save Prediction") and prediction is not None:
    log = input_df.copy()
    log["prediction"] = prediction
    log_file = "predictions.csv"
    log.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
    st.info("Prediction saved locally!")
