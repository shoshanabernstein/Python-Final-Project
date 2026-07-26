import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
with open("model.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
target_names = artifact["target_names"]
feature_names = artifact["feature_names"]

st.set_page_config(page_title="Obesity Classifier", layout="centered")

st.title("Obesity Classification App")
# Input widgets
col1, col2 = st.columns(2)

with col1:
    Age = st.slider("Age", min_value=14.0, max_value=61.0, value=25.0, step=1.0)

    Gender = st.segmented_control(
        "Gender",
        ["Male", "Female"]
    )

    CH2O = st.slider(
        "How much water do you drink daily",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.1
    )

    SCC = st.segmented_control(
        "Do you monitor the calories you eat daily",
        ["yes", "no"]
    )

    FAVC = st.segmented_control(
        "Do you eat high caloric food frequently",
        ["yes", "no"]
    )

    FCVC = st.slider(
        "Do you usually eat vegetables in your meals?",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.1
    )

    NCP = st.slider(
        "How many main meals do you have daily",
        min_value=1.0,
        max_value=4.0,
        value=3.0,
        step=0.1
    )


with col2:
    SMOKE = st.segmented_control(
        "Do you smoke?",
        ["yes", "no"]
    )

    FAF = st.slider(
        "How often do you have physical activity?",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1
    )

    TUE = st.slider(
        "How much time do you use technological devices daily?",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

    CALC = st.selectbox(
        "How often do you drink alcohol?",
        ["no", "Sometimes", "Frequently", "Always"]
    )

    MTRANS = st.selectbox(
        "What transportation do you usually use?",
        [
            "Public_Transportation",
            "Automobile",
            "Walking",
            "Motorbike",
            "Bike"
        ]
    )

    family_history_with_overweight = st.segmented_control(
        "Does your family have a history of being overweight?",
        ["yes", "no"]
    )

    CAEC = st.selectbox(
    "Do you eat any food between meals?",
    ["no", "Sometimes", "Frequently", "Always"]
)

input_data = pd.DataFrame([{
    "Gender": Gender,
    "Age": Age,
    "family_history_with_overweight": family_history_with_overweight,
    "FAVC": FAVC,
    "FCVC": FCVC,
    "NCP": NCP,
    "CAEC": CAEC,
    "SMOKE": SMOKE,
    "CH2O": CH2O,
    "SCC": SCC,
    "FAF": FAF,
    "TUE": TUE,
    "CALC": CALC,
    "MTRANS": MTRANS
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader("Prediction")
    st.success(
        f"Predicted: **{prediction.replace('_', ' ').title()}**"
    )

    st.subheader("Prediction Probabilities")

    for class_name, probability in zip(
        model.classes_,
        probabilities
    ):
        readable_name = class_name.replace("_", " ").title()
        st.write(f"{readable_name}: {probability:.2%}")
