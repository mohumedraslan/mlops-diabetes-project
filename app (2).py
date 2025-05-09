import streamlit as st
import joblib
import pandas as pd
import os

# Load model
model = joblib.load('diabetes_model_gbr.pkl')

st.title("Diabetes Progression Predictor")

# Sliders for user input (using realistic ranges from df.describe())
age = st.slider("Age (scaled)", -0.1, 0.1, 0.0)
sex = st.slider("Sex (scaled)", -0.1, 0.1, 0.0)
bmi = st.slider("BMI (scaled)", -0.1, 0.1, 0.0)
bp = st.slider("Blood Pressure (scaled)", -0.1, 0.1, 0.0)
s1 = st.slider("S1 (scaled)", -0.1, 0.1, 0.0)
s2 = st.slider("S2 (scaled)", -0.1, 0.1, 0.0)
s3 = st.slider("S3 (scaled)", -0.1, 0.1, 0.0)
s4 = st.slider("S4 (scaled)", -0.1, 0.1, 0.0)
s5 = st.slider("S5 (scaled)", -0.1, 0.1, 0.0)
s6 = st.slider("S6 (scaled)", -0.1, 0.1, 0.0)

# Create input data
input_data = pd.DataFrame([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]], 
                          columns=['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])

# Predict with error handling
try:
    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)  # Ensure non-negative
    st.write(f"Predicted Diabetes Progression: {prediction:.2f}")
except Exception as e:
    st.error(f"Error making prediction: {e}")

# Log prediction
if st.button("Save Prediction"):
    log = pd.DataFrame({"age": [age], "sex": [sex], "bmi": [bmi], "bp": [bp], 
                        "s1": [s1], "s2": [s2], "s3": [s3], "s4": [s4], 
                        "s5": [s5], "s6": [s6], "prediction": [prediction]})
    log_file = "predictions.csv"
    log.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
    st.write("Prediction saved!")
