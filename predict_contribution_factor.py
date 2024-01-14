import streamlit as st
import pickle
import numpy as np

def load_knn_model():
    with open('knn_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
    
data = load_knn_model()

knn_model = data['model']
borough_mapping = data['borough_mapping']
time_category_mapping = data['time_category_mapping']
reverse_contribution_factor_mapping = data['reverse_contribution_factor_mapping']
reverse_severity_mapping = data['reverse_severity_mapping']


def view_predict_contribution_factor_page():

    severities = (
        "Mild",
        "Moderate",
        "Severe"
    )

    boroughs = (
        "BROOKLYN",
        "QUEENS",
        "MANHATTAN",
        "BRONX",
        "STATEN ISLAND"
    )

    crash_times = (
        "Night",
        "Early-morning",
        "Afternoon-Evening"
    )
    
    severity = st.selectbox("Severity", severities)
    latitude = st.number_input("Latitude")
    longitude = st.number_input("Longitude")
    borough = st.selectbox("Borough", boroughs)
    crash_time = st.selectbox("Crash Time", crash_times)
    total_injured = st.number_input("Total Injured", min_value=0, step=1)
    total_killed = st.number_input("Total Killed", min_value=0, step=1)
    predict_contribution_factor = st.button("Predict Accident Cause")

    if predict_contribution_factor:
        X = np.array([[severity, latitude, longitude, borough, crash_time, total_injured, total_killed]])
        X[:,0] = np.vectorize(reverse_severity_mapping.get)(X[:,0])
        X[:,3] = np.vectorize(borough_mapping.get)(X[:,3])
        X[:,4] = np.vectorize(time_category_mapping.get)(X[:,4])
        X = X.astype(float)

        new_cf = knn_model.predict(X)
        result = reverse_contribution_factor_mapping.get(new_cf[0])

        st.subheader(f"The Accident is Most Likely Cause by: {result}")