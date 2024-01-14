import streamlit as st
from explore import view_explore_page
from predict_severity import view_predict_severity_page
from predict_contribution_factor import view_predict_contribution_factor_page

page = st.sidebar.selectbox("Predict Severity Predict Contribution Factor Explore", ("Predict Severity", "Predict Contribution Factor", "Explore"))

if(page == "Predict Severity"):
    view_predict_severity_page()
if(page == "Predict Contribution Factor"):
    view_predict_contribution_factor_page()
if(page == "Explore"):
    view_explore_page()