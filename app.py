import streamlit as st
import pandas as pd
import joblib
import numpy as np

df = pd.read_csv('model_df.csv')
st.set_page_config(page_title="Learning Mode Recommended for Students", layout="centered")

st.title("🎓 Learning Mode Recommended for Students for Each Academic Activity")

st.markdown("Fill in your profile below to get predictions for your preferred learning activities.")

# Define input features
single_select_features = ['gender', 'age', 'year_of_study', 'faculty',
                          'accommodation', 'commute_time', 'type_of_study',
                          'working', 'hours_work', 'internet',
                          'availability_quiet_space']

commute_modes = ['Driving', 'University bus', 'Grab/MyCar/Taxi', 'Public transport', 'Walking', 'Carpool']

user_input = {}

# Collect inputs
st.subheader("📝 General Profile")
for feature in single_select_features:
    options = sorted(df[feature].dropna().unique())
    user_input[feature] = st.selectbox(f"{feature.replace('_', ' ').title()}:", options)

st.subheader("🚌 Commute Mode (Select all that apply)")
selected_commute = st.multiselect("Commute Method:", commute_modes)

# One-hot encode commute modes
for mode in commute_modes:
    user_input[mode] = 1 if mode in selected_commute else 0

# Load models and make predictions
st.subheader("🔍 Predicted Preferences")

if st.button("Predict"):
    try:
        # Convert input to dataframe
        input_df = pd.DataFrame([user_input])

        # Load model filenames
        import os
        model_files = [f for f in os.listdir("tuned_models") if f.endswith(".pkl")]

        for model_file in model_files:
            activity_name = model_file.split("_")[0]
            model = joblib.load(f"tuned_models/{model_file}")
            prediction = model.predict(input_df)[0]
            st.write(f"**{activity_name}** Preference: {'In-Person' if prediction == 1 else 'Online'}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
