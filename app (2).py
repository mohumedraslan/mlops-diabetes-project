import streamlit as st
import joblib
import pandas as pd
import os

# Load trained model (no retraining)
model = joblib.load("diabetes_model_gbr.pkl")

st.title("Diabetes Progression Predictor")

st.markdown("""
⚠️ **Model Note**

This model was trained on the **standardized Diabetes dataset from scikit-learn**.
Inputs below are **z-score–normalized clinical features**, not raw medical values.
""")

# Feature ranges taken from training data (df.describe)
features = {
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

st.subheader("Standardized Feature Inputs")

for name, (min_val, max_val) in features.items():
    inputs[name] = st.slider(
        f"{name.upper()} (standardized)",
        float(min_val),
        float(max_val),
        0.0,
        step=0.001
    )

input_df = pd.DataFrame([inputs])

prediction = None

if st.button("Predict Progression"):
    prediction = model.predict(input_df)[0]
    prediction = max(0, prediction)

    st.success(f"Predicted Diabetes Progression Score: **{prediction:.2f}**")

    st.caption(
        "Prediction represents disease progression one year after baseline, "
        "as defined in the original diabetes dataset."
    )

if st.button("Save Prediction") and prediction is not None:
    log = input_df.copy()
    log["prediction"] = prediction

    log_file = "predictions.csv"
    log.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
    st.info("Prediction saved locally.")
