import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("sonar_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit App Configuration
st.set_page_config(page_title="Sonar Rock vs Mine Prediction", layout="wide")
st.title("🔍 Sonar Rock vs Mine Prediction App")
st.write("Enter the sonar readings or upload a CSV file to classify objects as **Rock** or **Mine**.")

# Option to Select Input Method
option = st.radio("Choose Input Method:", ["Manual Entry", "Upload CSV"])

if option == "Manual Entry":
    # User enters a single row of 60 features as comma-separated values
    user_input = st.text_area("Enter 60 comma-separated values:", "0.07,0.12,0.14,0.16,0.16,0.08,0.07,0.09,0.15,0.12,0.23,0.17,0.22,0.31,0.47,0.55,0.53,0.37,0.45,0.59,0.74,0.76,0.57,0.62,0.66,0.71,0.75,0.80,0.84,0.92,0.95,0.79,0.43,0.40,0.49,0.17,0.36,0.37,0.28,0.04,0.53,0.69,0.42,0.30,0.58,0.56,0.26,0.11,0.09,0.03,0.02,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.02")
    
    # Convert input to numpy array
    try:
        input_data = np.array([float(x) for x in user_input.split(",")]).reshape(1, -1)
        valid_input = True
    except:
        st.error("❌ Please enter 60 valid comma-separated numbers.")
        valid_input = False

elif option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with 60 features per row", type=["csv"])
    if uploaded_file:
        import pandas as pd
        df = pd.read_csv(uploaded_file, header=None)
        input_data = df.values
        valid_input = True
    else:
        valid_input = False

# Predict Button
if st.button("🚀 Predict") and valid_input:
    prediction = model.predict(input_data)
    result = ["🪨 Rock" if p == 'R' else "🛠️ Mine" for p in prediction]

    # Show results
    if len(result) == 1:
        st.success(f"**Prediction:** The object is a **{result[0]}**")
    else:
        st.success("**Batch Predictions:**")
        st.write(result)
