import streamlit as st
import joblib
import pandas as pd
import os

# Load model
model = joblib.load("diabetes_model_gbr.pkl")

st.title("Diabetes Progression Predictor")

st.markdown(
    """
    **Note:** Inputs are *standardized values* (same scale used during model training).
    """
)

# Feature ranges from df.describe()
ranges = {
    "age": (-0.11, 0.11),
    "sex": (-0.045, 0.051),
    "bmi": (-0.09, 0.17),
    "bp": (-0.11, 0.13),
    "s1": (-0.13, 0.15),
    "s2": (-0.12, 0.20),
    "s3": (-0.10, 0.18),
    "s4": (-0.076, 0.185),
    "s5": (-0.13, 0.134),
    "s6": (-0.14, 0.136),
}

inputs = {}
for feature, (min_val, max_val) in ranges.items():
    inputs[feature] = st.slider(
        f"{feature.upper()} (scaled)",
        min_value=float(min_val),
        max_value=float(max_val),
        value=0.0,
        step=0.001,
    )

# Create input DataFrame
input_df = pd.DataFrame([inputs])

prediction = None

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction = max(0, prediction)
    st.success(f"Predicted Diabetes Progression: **{prediction:.2f}**")

# Save prediction safely
if st.button("Save Prediction") and prediction is not None:
    log = input_df.copy()
    log["prediction"] = prediction

    log_file = "predictions.csv"
    log.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
    st.write("Prediction saved!")
