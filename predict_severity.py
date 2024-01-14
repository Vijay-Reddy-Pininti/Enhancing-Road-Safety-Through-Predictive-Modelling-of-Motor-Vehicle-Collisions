import streamlit as st
import pickle
import numpy as np

def load_nb_model():
    with open('naive_bayes_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
    
data = load_nb_model()

nb_model = data["model"]
borough_mapping = data["borough_mapping"]
time_category_mapping = data["time_category_mapping"]
contribution_factor_encoder = data["contribution_factor_encoder"]
factor_mapping = data["factor_mapping"]
severity_mapping = data["severity_mapping"]


def view_predict_severity_page():
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

    contribution_factors = (
        'UNSPECIFIED', 
        'DRIVER INEXPERIENCE', 
        'PASSING TOO CLOSELY', 
        'TURNING IMPROPERLY', 
        'REACTION TO UNINVOLVED VEHICLE', 
        'STEERING FAILURE', 
        'FOLLOWING TOO CLOSELY', 
        'PASSING OR LANE USAGE IMPROPER', 
        'DRIVER INATTENTION/DISTRACTION', 
        'OVERSIZED VEHICLE', 
        'UNSAFE LANE CHANGING', 
        'ALCOHOL INVOLVEMENT', 
        'VIEW OBSTRUCTED/LIMITED', 
        'TRAFFIC CONTROL DISREGARDED', 
        'FAILURE TO YIELD RIGHT-OF-WAY', 
        'AGGRESSIVE DRIVING/ROAD RAGE', 
        'UNSAFE SPEED', 
        'PAVEMENT SLIPPERY', 
        'ILLNES', 
        'LOST CONSCIOUSNESS', 
        'OTHER VEHICULAR', 
        'BRAKES DEFECTIVE', 
        'BACKING UNSAFELY', 
        'PASSENGER DISTRACTION', 
        'FELL ASLEEP', 
        'OBSTRUCTION/DEBRIS', 
        'TINTED WINDOWS', 
        'PEDESTRIAN/BICYCLIST/OTHER PEDESTRIAN ERROR/CONFUSION', 
        'ANIMALS ACTION', 
        'DRUGS (ILLEGAL)', 
        'OUTSIDE CAR DISTRACTION', 
        'TIRE FAILURE/INADEQUATE', 
        'PAVEMENT DEFECTIVE', 
        'FATIGUED/DROWSY', 
        'ACCELERATOR DEFECTIVE', 
        'PHYSICAL DISABILITY', 
        'GLARE', 
        'DRIVERLESS/RUNAWAY VEHICLE', 
        'EATING OR DRINKING', 
        'FAILURE TO KEEP RIGHT', 
        'CELL PHONE (HANDS-FREE)', 
        'LANE MARKING IMPROPER/INADEQUATE', 
        'HEADLIGHTS DEFECTIVE', 
        'CELL PHONE (HAND-HELD)', 
        'WINDSHIELD INADEQUATE', 
        'VEHICLE VANDALISM', 
        'USING ON BOARD NAVIGATION DEVICE', 
        'PRESCRIPTION MEDICATION', 
        'OTHER ELECTRONIC DEVICE', 
        'TRAFFIC CONTROL DEVICE IMPROPER/NON-WORKING', 
        'TEXTING', 
        'TOW HITCH DEFECTIVE', 
        'OTHER LIGHTING DEFECTS', 
        'SHOULDERS DEFECTIVE/IMPROPER', 
        'LISTENING/USING HEADPHONES', 
        'REACTION TO OTHER UNINVOLVED VEHICLE', 
        '80', 
        '1', 
        'ILLNESS'
    )


    borough = st.selectbox("Borough", boroughs)
    crash_time = st.selectbox("Crash Time", crash_times)
    contribution = st.selectbox("Contribution Factor", contribution_factors)
    total_injured = st.number_input("Total Injured", min_value=0, step=1)
    total_killed = st.number_input("Total Killed", min_value=0, step=1)
    predict_severity = st.button("Predict Severity")

    if predict_severity:
        X = np.array([[borough, crash_time, total_injured, total_killed, contribution]])
        X[:,0] = np.vectorize(borough_mapping.get)(X[:,0])
        X[:,1] = np.vectorize(time_category_mapping.get)(X[:,1])
        X[:,4] = np.vectorize(factor_mapping.get)(X[:,4])
        X[:,4] = np.vectorize(contribution_factor_encoder.get)(X[:,4])
        X = X.astype(float)

        new_severity = nb_model.predict(X)
        result = severity_mapping.get(new_severity[0])

        st.subheader(f"The Predicted Severity is: {result}")