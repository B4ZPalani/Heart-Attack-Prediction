import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import pickle
import sklearn




def main():
    @st.cache(persist=True)
   


    st.set_page_config(
        page_title="Heart Disease Prediction App",
        page_icon="images/heart-fav.png"
    )

    st.title("Heart Disease Prediction")
    st.subheader("Are you wondering about the condition of your heart? "
                 "This app will help you to diagnose it!")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("images/doctor.png",
                 caption="I'll help you diagnose your heart health! - Dr. Logistic Regression",
                 width=150)
        submit = st.button("Predict")
    with col2:
    
    st.sidebar.title("Feature Selection")
    st.sidebar.image("images/heart-sidebar.png", width=100)

 
if __name__ == "__main__":
    main()
