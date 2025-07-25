import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="Fetal Health Prediction", layout="wide")

# Sidebar Menu
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:", ["Home", "Fetal Health Prediction", "About project"])

# -------- Page 1: HOME --------
if option == "Home":
    st.title("Welcome to the Fetal Health Prediction App 👶")
    st.subheader("Your step towards healthier pregnancies")
    
    # Image 
    st.image("https://preganews.com/wp-content/uploads/2023/04/goD3K4n2Xi4w1tZIr1Qp.png", caption="Fetal Health Awareness", use_container_width=500)

    st.markdown("""
    ###  What is Fetal Health?
    Fetal health refers to the well-being of a fetus during pregnancy. Healthy development ensures the baby grows properly and avoids complications at birth.

    ### ✅ Prevention Tips:
    - Regular prenatal check-ups
    - Healthy diet and hydration
    - Avoid smoking, alcohol, and drugs
    - Proper rest and stress control

    ---
    ### 🔍 Click on the sidebar to predict fetal health using your data.
    """)

# ----------- FETAL HEALTH PREDICTION PAGE -----------
elif option == "Fetal Health Prediction":
    st.title("🩺 Fetal Heralth Prediction App")
    st.write("Enter the fetal diagnostic features below to predict the health status.")
    # Load the model
    model_path = "save_models/fetal_health.joblib"
    try:
        loaded_model = joblib.load(model_path)
        
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}")
        st.stop()



    # Input Features (21 total)
    features = {
    "baseline value": st.number_input("Baseline Value", 50, 200, 120),
    "accelerations": st.number_input("Accelerations", 0.0, 1.0),
    "fetal_movement": st.number_input("Fetal Movement", 0.0, 1.0, 0.0),
    "uterine_contractions": st.number_input("Uterine Contractions", 0.0, 1.0, 0.0),
    "light_decelerations": st.number_input("Light Decelerations", 0.0, 1.0, 0.0),
    "severe_decelerations": st.number_input("Severe Decelerations", 0.0, 1.0, 0.0),
    "prolongued_decelerations": st.number_input("Prolonged Decelerations", 0.0, 1.0, 0.0),
    "abnormal_short_term_variability": st.number_input("Abnormal Short Term Variability", 0, 100),
    "mean_value_of_short_term_variability": st.number_input("Mean Short Term Variability", 0.0, 10.0, 0.5),
    "percentage_of_time_with_abnormal_long_term_variability": st.number_input("Abnormal Long Term Variability (%)", 0.0, 100.0, 0.0),
    "mean_value_of_long_term_variability": st.number_input("Mean Long Term Variability", 0.0, 50.0, 5.0),
    "histogram_width": st.number_input("Histogram Width", 0, 200, 50),
    "histogram_min": st.number_input("Histogram Min", 0, 150, 0),
    "histogram_max": st.number_input("Histogram Max", -50, 200, 50),
    "histogram_number_of_peaks": st.number_input("Histogram Peaks", 0, 20, 2),
    "histogram_number_of_zeroes": st.number_input("Histogram Zeroes", 0, 20, 2),
    "histogram_mode": st.number_input("Histogram Mode", 0, 200, 120),
    "histogram_mean": st.number_input("Histogram Mean", 0, 200, 120),
    "histogram_median": st.number_input("Histogram Median", 0, 200, 120),
    "histogram_variance": st.number_input("Histogram Variance", 0, 1000, 100),
    "histogram_tendency": st.number_input("Histogram Tendency", -10, 10, 0)
    }

    # Prediction
    if st.button("Predict Fetal Health"):
        input_data = np.array([list(features.values())])
        prediction = loaded_model.predict(input_data)[0]

        # Label map
        label_map = {
        0: "🟢 Normal - The baby appears healthy.",
        1: "🟡 Suspect - Signs of possible concern. Please consult a doctor.",
        2: "🔴 Pathological - The baby is in danger. Immediate medical attention is required."
        }

        st.subheader("Prediction Result:")
        st.write(f"## {label_map.get(prediction, '⚠️ Unknown Result')}")

# ----------- ABOUT ME PAGE -----------
elif option == "About project":
    # About Project Page
    st.set_page_config(page_title="About Project", layout="centered")

    st.title(" About the Fetal Health Prediction Project")

    st.markdown("---")

    # Project Overview
    st.header(" Project Overview")
    st.markdown("""
    Fetal health monitoring is a crucial aspect of prenatal care that helps ensure the healthy development of a fetus during pregnancy. Traditional monitoring methods require continuous expert analysis and often delay early diagnosis of complications.

     """)

    # Problem Statement
    st.subheader("Problem Statement")
    st.markdown("""
    The lack of automated, real-time analysis tools for fetal health data can lead to late detection of risks during pregnancy, potentially endangering the life of both the fetus and the mother.
    """)

    # Solution
    st.subheader("✅ Proposed Solution")
    st.markdown("""This project proposes a machine learning-based solution that predicts fetal health using cardiotocography (CTG) data. The system classifies fetal states as **Normal**, **Suspect**, or **Pathological**, enabling timely medical intervention.""")
    # Features Considered
    st.subheader("Features Considered")
    st.markdown("""
    - Baseline Value  
                
    - Accelerations  
    - Fetal Movement  
    - Uterine Contractions  
    - Light/Severe/Prolonged Decelerations  
    - Short and Long Term Variability  
    - Histogram Statistics (Width, Min, Max, Mode, Mean, Median, Variance, etc.)
    """)
     # Steps Performed
    st.subheader(" Steps Performed")
    st.markdown("""
    1. Data Collection and Preprocessing  
    2. Feature Selection  
    3. Model Training using Machine Learning Algorithms  
    4. Evaluation and Optimization  
    5. Streamlit App Development  
    6. Model Deployment using joblib to save and load trained model
    """)

    # Deployment
    st.subheader("Deployment")
    st.markdown("""
    This web-based machine learning app is deployed live on the cloud using tool like **Render** making it accessible for real-time fetal health prediction.
    """)

    # Submission Info
    st.markdown("### Submitted By")
    st.markdown("""
    **Komalpreet Kaur**  
    **B.Tech in Electronics and Computer Engineering**
    **Guru Nanak Dev University  Amritsar**
    """)

    # Skills & Achievements
    st.subheader("Skills and Achievements")

    st.markdown("**Technical Skills:**")
    st.markdown("""
    - **Languages:** C, C++, Python  
    - **Domains:** IoT, Machine Learning  
    - **Tools & Platforms:** Streamlit, Git, GitHub
    """)

    st.markdown("**🌟 Other Achievements:**")
    st.markdown("""
    - Performed **choreography** at **Jashan fest in GNDU ** and won **second prize**
    """)

    st.markdown("### Thank You for Visiting the App!")

    st.markdown("""
    If you face any issues or feel the fetal health prediction is not accurate,  
    please **consult a doctor immediately**.  

    Always follow medical advice, get the necessary tests done,  
    and take care of yourself and your baby’s health.  

    **Your well-being is our priority. ❤️**
    """)
