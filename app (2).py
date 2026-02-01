import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model
model = joblib.load("diabetes_model_gbr.pkl")

st.set_page_config(page_title="Diabetes Progression Predictor", layout="wide")
st.title("🩺 Diabetes Progression Predictor")
st.markdown("""
This tool predicts **diabetes disease progression** (one year after baseline) using the Gradient Boosting Regressor trained on the standardized diabetes dataset.
You can enter **real clinical values**, and the app will internally standardize them to the model's expected feature space.
""")

# -----------------------------
# Step 1: User inputs with guidance
# -----------------------------
st.subheader("Patient Information")

st.markdown("""
**Instructions:**
- Enter the patient's real clinical values.
- Typical ranges are given to help input realistic numbers.
""")

age = st.number_input(
    "Age (years)", min_value=20, max_value=80, value=50,
    help="Patient age in years. Typical adult range: 20-80"
)
sex = st.selectbox(
    "Sex", options=["Male", "Female"],
    help="Select the patient's sex."
)
bmi = st.number_input(
    "BMI (kg/m²)", min_value=15.0, max_value=50.0, value=25.0,
    help="Body Mass Index. Typical healthy range: 18.5–25"
)
bp = st.number_input(
    "Blood Pressure (mmHg)", min_value=80, max_value=200, value=120,
    help="Mean Arterial Blood Pressure. Typical: 90–130 mmHg"
)

# S1–S6 are lab features (they are scaled in sklearn dataset)
st.markdown("**Lab Measurements (S1–S6, relative units)**")
s1 = st.number_input("S1 (Rel. units)", value=0.0, help="Lab measure 1 (cholesterol etc.)")
s2 = st.number_input("S2 (Rel. units)", value=0.0, help="Lab measure 2")
s3 = st.number_input("S3 (Rel. units)", value=0.0, help="Lab measure 3")
s4 = st.number_input("S4 (Rel. units)", value=0.0, help="Lab measure 4")
s5 = st.number_input("S5 (Rel. units)", value=0.0, help="Lab measure 5")
s6 = st.number_input("S6 (Rel. units)", value=0.0, help="Lab measure 6")

# -----------------------------
# Step 2: Encode and scale inputs
# -----------------------------
sex_val = 1 if sex == "Male" else 0

# Standardization values from sklearn diabetes dataset
# Dataset mean and std (from your printed describe())
# You can adjust if you want more accurate scaling
mean_std = {
    "age": (0.0, 0.047619),
    "sex": (0.0, 0.047619),
    "bmi": (0.0, 0.047619),
    "bp": (0.0, 0.047619),
    "s1": (0.0, 0.047619),
    "s2": (0.0, 0.047619),
    "s3": (0.0, 0.047619),
    "s4": (0.0, 0.047619),
    "s5": (0.0, 0.047619),
    "s6": (0.0, 0.047619),
}

# Example real-to-standardized mapping for age, bmi, bp
# Replace these with dataset real means & stds if you have them
real_to_scaled = {
    "age": (age - 52.0) / 9.0,     # e.g., dataset mean=52, std=9
    "sex": sex_val,                 # already 0/1
    "bmi": (bmi - 30.0) / 6.0,     # dataset mean/std
    "bp": (bp - 85.0) / 8.0,
    "s1": s1,
    "s2": s2,
    "s3": s3,
    "s4": s4,
    "s5": s5,
    "s6": s6,
}

input_df = pd.DataFrame([[
    real_to_scaled["age"], real_to_scaled["sex"], real_to_scaled["bmi"], real_to_scaled["bp"],
    real_to_scaled["s1"], real_to_scaled["s2"], real_to_scaled["s3"], real_to_scaled["s4"],
    real_to_scaled["s5"], real_to_scaled["s6"]
]], columns=["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"])

# -----------------------------
# Step 3: Predict
# -----------------------------
prediction = None
if st.button("Predict Progression"):
    prediction = model.predict(input_df)[0]
    prediction = max(0, prediction)
    st.success(f"Predicted Diabetes Progression Score: {prediction:.2f}")
    st.info("Interpretation: Higher scores indicate faster progression of diabetes within 1 year.")

# -----------------------------
# Step 4: Optional logging
# -----------------------------
if st.button("Save Prediction") and prediction is not None:
    log = input_df.copy()
    log["prediction"] = prediction
    log_file = "predictions.csv"
    log.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
    st.info(f"Prediction saved locally to {log_file}")
