import streamlit as st
import pickle
import numpy as np

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
    Age = st.slider("Age", min_value=0.0, max_value=120.0, value=1.0, step=0.1)
    Gender = st.segmented_control("Gender", ['Male', 'Female'])
    CH2O = st.slider("How much water do you drink daily", min_value=1.3, max_value=3.3, value=2.4, step=0.1)
    SCC = st.slider("Do you monitor the calories you eat daily", min_value=10.0, max_value=30.0, value=19.5, step=0.5)
    FAVC = st.slider("Do you eat high caloric food frequently", min_value=70.0, max_value=165.0, value=100.0, step=1.0)
    FCVC = st.slider("Do you usually eat vegetables in your meals?", min_value=0.9, max_value=4.0, value=2.3, step=0.1)
    NCP = st.slider("How many main meals do you have daily", min_value=0.3, max_value=5.1, value=2.0, step=0.1)

with col2:
    SMOKE = st.slider("Do you smoke? ", min_value=0.1, max_value=0.7, value=0.4, step=0.05)
    FAF = st.slider("How often do you have physical activity?", min_value=0.4, max_value=3.6, value=1.6, step=0.1)
    TUE = st.slider("How much time do you use technological devices daily?", min_value=1.2, max_value=13.0, value=5.0, step=0.1)
    CALC = st.slider("How often do you drink alcohol? ", min_value=0.4, max_value=1.8, value=1.0, step=0.05)
    MTRANS = st.slider("What transportation do you usually use? ", min_value=1.2, max_value=4.0, value=2.6, step=0.1)
    family_history_with_overweight = st.slider("Proline", min_value=270.0, max_value=1680.0, value=745.0, step=10.0)

input_data = np.array([[
    Age, Gender, CH2O, SCC, family_history_with_overweight, FAVC, 
    FCVC, NCP, SMOKE, 
    FAF, TUE, CALC, 
    MTRANS
]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    predicted_class_name = target_names[prediction].replace("_", " ").title()

    st.subheader("Prediction")
    st.success(f"Predicted: **{predicted_class_name}**")

    st.subheader("Prediction probabilities")
    for i, class_name in enumerate(target_names):
        st.write(f"{class_name}: {probabilities[i]:.2%}")