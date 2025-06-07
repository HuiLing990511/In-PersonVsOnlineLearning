import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('model_df.csv')
df['hours_work'] = df['hours_work'].replace('', 'No working').fillna('No working')

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
    options = sorted(df[feature].unique())
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

        input_df['age'] = input_df['age'].map({
            '15 - 19': 0,
            '20 - 24': 1,
            '25 - 29': 2,
            '30 - 34': 3,
            '35 - 39' : 4,
            '40 - 44' : 5, 
            '45 - 49': 6,
            '49 and above' : 7
        })

        input_df['year_of_study'] = input_df['year_of_study'].map({
            'Foundation': 0,
            '1st Year': 1,
            '2nd Year': 2,
            '3rd Year': 3,
            '4th Year' : 4,
            '5th year' : 5, 
            'Postgraduate (Masters/PhD)': 6,
        })

        input_df['accommodation'] = input_df['accommodation'].map({
            'Off-campus': 1,
            'Hostel': 0
        })

        input_df['commute_time'] = input_df['commute_time'].map({
            'Under 15 min': 0,
            '15 min to 30 min': 1,
            '30 min to 1 hr': 2,
            '1 hr+': 3
        })

        input_df['type_of_study'] = input_df['type_of_study'].map({
            'Part-time student': 1,
            'Full-time student': 0
        })

        input_df['working'] = input_df['working'].map({
            'Yes': 1,
            'No': 0
        })

        input_df['hours_work'] = input_df['hours_work'].map({
            'No working': 0,
            'Less than 10 hours per week': 1,
            '10 to 30 hours per week': 2,
            'More than 30 hours per week': 3
        })

        input_df['availability_quiet_space'] = input_df['availability_quiet_space'].map({
            'No, never': 0,
            'No, rarely': 1,
            'Yes, sometimes': 2,
            'Yes, consistently': 3
        })

        le = LabelEncoder()

        # Encode categorical features
        categorical_cols = ['gender', 'faculty']
        for col in categorical_cols:
            input_df[col] = le.fit_transform(input_df[col].astype(str))

        # Load model filenames
        import os
        model_files = [f for f in os.listdir("tuned_models") if f.endswith(".pkl")]

        for model_file in model_files:
            activity_name = model_file.split("_")[0]
            model = joblib.load(f"tuned_models/{model_file}")
            prediction = model.predict(input_df)[0]
            st.markdown(f"**⭐{activity_name}**       : {'In-Person' if prediction == 0 else 'Online'}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
