import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('model_df.csv')
df['hours_work'] = df['hours_work'].replace('','No working')
st.set_page_config(page_title="Learning Mode Recommended for Students", layout="centered")

st.title("🎓 Personalized Learning Mode Recommendations for Academic Activities")

st.markdown("Fill in the student profile to get personalized study mode recommendations!")

# Define input features
single_select_features = ['gender', 'age', 'year_of_study', 'faculty',
                          'accommodation', 'commute_time', 'type_of_study',
                          'working', 'hours_work', 'internet',
                          'availability_quiet_space'
                          ]

commute_modes = ['Driving', 'University bus', 'Grab/MyCar/Taxi', 'Public transport', 'Walking', 'Carpool']

user_input = {}

# Collect inputs
st.subheader("📝 General Profile")
for feature in single_select_features:
    
    options = sorted(df[feature].dropna().unique())

    # Custom label for internet feature
    if feature == 'internet':
        label = "Internet Connectivity (1 = Poor, 4 = Good):"
    else:
        label = f"{feature.replace('_', ' ').title()}:"

    user_input[feature] = st.selectbox(label, options)

st.subheader("🚌 Commute Mode (Select all that apply)")
selected_commute = st.multiselect("Commute Method:", commute_modes)

# One-hot encode commute modes
for mode in commute_modes:
    user_input[mode] = 1 if mode in selected_commute else 0

# Load models and make predictions
st.subheader("🔍 Personalized Learning Mode for Each Academic Activity")

if st.button("Recommend"):
    try:
        # Convert input to dataframe
        input_df = pd.DataFrame([user_input])
        le = LabelEncoder()

        # Encode categorical features
        categorical_cols = ['gender', 'age', 'year_of_study', 'faculty', 'accommodation', 'commute_time',
                            'type_of_study', 'working', 'hours_work', 'internet',
                            'availability_quiet_space']

        for col in categorical_cols:
            input_df[col] = le.fit_transform(input_df[col].astype(str))

        # Load model filenames
        import os
        model_files = [f for f in os.listdir("tuned_models") if f.endswith(".pkl")]

        for model_file in model_files:
            activity_name = model_file.split("_")[0]
            model = joblib.load(f"tuned_models/{model_file}")
            prediction = model.predict(input_df)[0]
            st.markdown(f"**⭐{activity_name}**       : {'In-Person' if prediction == 1 else 'Online'}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
