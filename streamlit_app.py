import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

# Set page configuration outside of any function
st.set_page_config(
    page_title="ByteForza - Heart Attack Prediction App",
    page_icon="images/heart-fav.png"
)

# The main function
def main():
    # Cache the results of expensive operations with st.cache
    @st.cache(persist=True)
    def load_data():
        # Placeholder for any data loading or processing logic
        pass

    st.title("Heart Attack Prediction1")
    st.subheader("Concerned about your heart health? This app is here to help you assess your risk and take proactive steps to safeguard your heart!")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("images/aidocter.jpg",
                 caption="I'll help you diagnose your heart health! - Dr. ByteForza AI",
                 width=150)
        submit = st.button("Predict")
    with col2:
        st.markdown("""
        Did you know that machine learning models can help you
        predict heart disease pretty accurately? In this app, you can
        estimate your chance of heart disease (yes/no) in seconds!
        
        Here, a logistic regression model using an undersampling technique
        was constructed using survey data of over 300k US residents from the year 2020.
        This application is based on it because it has proven to be better than the random forest
        (it achieves an accuracy of about 80%, which is quite good).
        
        To predict your heart disease status, simply follow the steps below:
        1. Enter the parameters that best describe you;
        2. Press the "Predict" button and wait for the result.
        """)

    st.sidebar.title("Feature Selection")
    st.sidebar.image("images/heart-sidebar.png", width=100)

if __name__ == "__main__":
    main()
