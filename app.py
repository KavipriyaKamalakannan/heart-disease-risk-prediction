
import streamlit as st
import pandas as pd
import joblib

# Load the trained machine learning model
model = joblib.load("heart_disease_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

# Application title
st.title("❤️ Heart Disease Risk Prediction")

st.write(
    "Enter the clinical parameters below "
    "to obtain a machine learning classification."
)

st.info(
    "Educational prototype only. "
    "This application is not a medical diagnostic tool."
)

# Patient information
st.header("Patient Clinical Information")

age = st.number_input(
    "Age",
    min_value=20,
    max_value=100,
    value=55
)

sex = st.selectbox(
    "Sex (0 or 1)",
    [0, 1]
)

cp = st.selectbox(
    "Chest Pain Type",
    [1, 2, 3, 4]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=300,
    value=130
)

chol = st.number_input(
    "Serum Cholesterol",
    min_value=50,
    max_value=700,
    value=240
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

restecg = st.selectbox(
    "Resting ECG",
    [0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=40,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=1.2,
    step=0.1
)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    [1, 2, 3]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3]
)

thal = st.selectbox(
    "Thalassemia Category",
    [3, 6, 7]
)

# Prediction button
if st.button("Predict Heart Disease Class"):

    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    prediction = model.predict(patient)[0]

    probability = model.predict_proba(patient)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Heart disease class predicted")
    else:
        st.success("No heart disease class predicted")

    st.write(
        "Model-estimated probability of class 1:",
        f"{probability * 100:.2f}%"
    )

    st.caption(
        "This result is an experimental machine learning "
        "classification, not a clinical risk assessment "
        "or medical diagnosis."
    )
