import streamlit as st
import pickle
import numpy as np

# Load models
with open('/mnt/data/parkinsons.pkl', 'rb') as file:
    parkinsons_model = pickle.load(file)

with open('/mnt/data/kidney.pkl', 'rb') as file:
    kidney_model = pickle.load(file)

with open('/mnt/data/liver.pkl', 'rb') as file:
    liver_model = pickle.load(file)

# Define prediction functions
def predict_parkinsons(features):
    prediction = parkinsons_model.predict([features])
    return 'Disease Detected' if prediction[0] == 1 else 'No Disease'

def predict_kidney(features):
    prediction = kidney_model.predict([features])
    return 'Disease Detected' if prediction[0] == 1 else 'No Disease'

def predict_liver(features):
    prediction = liver_model.predict([features])
    return 'Disease Detected' if prediction[0] == 1 else 'No Disease'

# Streamlit app
st.title('Disease Prediction App')

# Sidebar for disease selection
disease = st.sidebar.selectbox(
    'Select a Disease to Predict:',
    ('Parkinsons', 'Kidney', 'Liver')
)

# Input fields and prediction logic
if disease == 'Parkinsons':
    st.header('Parkinsons Prediction')
    mdvp_fo = st.number_input('MDVP:Fo(Hz)', value=0.0)
    mdvp_fhi = st.number_input('MDVP:Fhi(Hz)', value=0.0)
    mdvp_flo = st.number_input('MDVP:Flo(Hz)', value=0.0)
    mdvp_jitter_percent = st.number_input('MDVP:Jitter(%)', value=0.0)
    shimmer = st.number_input('Shimmer', value=0.0)
    rpde = st.number_input('RPDE', value=0.0)

    if st.button('Predict Parkinsons'):
        features = [mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent, shimmer, rpde]
        result = predict_parkinsons(features)
        st.success(result)

elif disease == 'Kidney':
    st.header('Kidney Disease Prediction')
    age = st.number_input('Age', value=0)
    bp = st.number_input('Blood Pressure', value=0)
    sg = st.number_input('Specific Gravity', value=0.0)
    al = st.number_input('Albumin', value=0)
    su = st.number_input('Sugar', value=0)

    if st.button('Predict Kidney Disease'):
        features = [age, bp, sg, al, su]
        result = predict_kidney(features)
        st.success(result)

elif disease == 'Liver':
    st.header('Liver Disease Prediction')
    age = st.number_input('Age', value=0)
    gender = st.selectbox('Gender', ('Male', 'Female'))
    tb = st.number_input('Total Bilirubin', value=0.0)
    db = st.number_input('Direct Bilirubin', value=0.0)
    alkphos = st.number_input('Alkaline Phosphotase', value=0)

    if st.button('Predict Liver Disease'):
        gender_binary = 1 if gender == 'Male' else 0
        features = [age, gender_binary, tb, db, alkphos]
        result = predict_liver(features)
        st.success(result)

st.write("\n")
st.info("Note: This is a demo app. Ensure you input accurate data for better predictions.")

