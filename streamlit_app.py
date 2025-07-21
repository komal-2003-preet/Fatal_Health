import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("fetal_health.csv")  # make sure this CSV is in the same folder

# Select only 9 important features
features = [
    "baseline value", "accelerations", "fetal_movement",
    "uterine_contractions", "light_decelerations", "severe_decelerations",
    "prolongued_decelerations", "abnormal_short_term_variability",
    "mean_value_of_short_term_variability"
]

X = df[features]
y = df["fetal_health"]

# Train and save 4 models
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVC": SVC(),
    "Decision Tree": DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X, y)
    with open(f"{name.replace(' ', '_')}.pkl", "wb") as f:
        pickle.dump(model, f)

import streamlit as st
import numpy as np
import pickle

def load_model(model_name):
    file_name = f"{model_name.replace(' ', '_')}.pkl"
    with open(file_name, "rb") as f:
        return pickle.load(f)

st.set_page_config(page_title="Fetal Heart Disease Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: teal;'>🩺 Fetal Heart Disease Prediction</h1>", unsafe_allow_html=True)

st.sidebar.image("https://cdn.pixabay.com/photo/2014/04/03/11/50/heart-312013_960_720.png", width=150)
model_choice = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "Logistic Regression", "SVC", "Decision Tree"])

with st.form("fetal_form"):
    st.subheader("🔍 Enter the test parameters:")
    col1, col2, col3 = st.columns(3)

    with col1:
        baseline_value = st.number_input("Baseline Value", 50, 200, 120)
        accelerations = st.number_input("Accelerations", 0.00, 1.00, 0.02)
        fetal_movement = st.number_input("Fetal Movement", 0.0, 1.0, 0.05)

    with col2:
        uterine_contractions = st.number_input("Uterine Contractions", 0.0, 1.0, 0.03)
        light_decelerations = st.number_input("Light Decelerations", 0.0, 1.0, 0.01)
        severe_decelerations = st.number_input("Severe Decelerations", 0.0, 1.0, 0.0)

    with col3:
        prolongued_decelerations = st.number_input("Prolongued Decelerations", 0.0, 1.0, 0.0)
        abnormal_short_term_variability = st.slider("Abnormal Short Term Variability", 0, 10, 0)
        mean_value_of_short_term_variability = st.slider("Mean STV", 0.0, 10.0, 1.5)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = np.array([[
        baseline_value, accelerations, fetal_movement,
        uterine_contractions, light_decelerations, severe_decelerations,
        prolongued_decelerations, abnormal_short_term_variability,
        mean_value_of_short_term_variability
    ]])

    model = load_model(model_choice)
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Prediction: Healthy")
    elif prediction == 2:
        st.warning("⚠️ Prediction: Suspect")
    else:
        st.error("🚨 Prediction: Pathological")
