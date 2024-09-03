import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

DATASET_PATH = "data/heart_2020_cleaned.csv"

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


df = pd.read_csv(DATASET_PATH);
df.drop(['Stroke']);
X = df.drop('HeartDisease',axis=1);
y = df['HeartDisease'];

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
# Initialize and train the Logistic Regression model
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
# Initialize and train the RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Load the dataset
@st.cache_resource
def load_dataset() -> pd.DataFrame:
    heart_df = pd.read_csv(DATASET_PATH)
    heart_df = pd.DataFrame(np.sort(heart_df.values, axis=0),
                            index=heart_df.index,
                            columns=heart_df.columns)
    return heart_df

# User input features function
def user_input_features(heart: pd.DataFrame) -> pd.DataFrame:
    sex = st.sidebar.selectbox("Sex", options=heart['Sex'].unique())
    age_cat = st.sidebar.selectbox("Age category", options=heart['AgeCategory'].unique())
    bmi_cat = st.sidebar.selectbox("BMI category", options=heart['BMICategory'].unique())
    sleep_time = st.sidebar.number_input("How many hours on average do you sleep?", 0, 24, 7)
    gen_health = st.sidebar.selectbox("How can you define your general health?", options=heart['GenHealth'].unique())
    phys_health = st.sidebar.number_input("For how many days during the past 30 days was your physical health not good?", 0, 30, 0)
    ment_health = st.sidebar.number_input("For how many days during the past 30 days was your mental health not good?", 0, 30, 0)
    phys_act = st.sidebar.selectbox("Have you played any sports (running, biking, etc.) in the past month?", options=("No", "Yes"))
    smoking = st.sidebar.selectbox("Have you smoked at least 100 cigarettes in your entire life (approx. 5 packs)?", options=("No", "Yes"))
    alcohol_drink = st.sidebar.selectbox("Do you have more than 14 drinks of alcohol (men) or more than 7 (women) in a week?", options=("No", "Yes"))
    diff_walk = st.sidebar.selectbox("Do you have serious difficulty walking or climbing stairs?", options=("No", "Yes"))
    diabetic = st.sidebar.selectbox("Have you ever had diabetes?", options=heart['Diabetic'].unique())
    asthma = st.sidebar.selectbox("Do you have asthma?", options=("No", "Yes"))
    kid_dis = st.sidebar.selectbox("Do you have kidney disease?", options=("No", "Yes"))
    skin_canc = st.sidebar.selectbox("Do you have skin cancer?", options=("No", "Yes"))

    features = pd.DataFrame({
        "PhysicalHealth": [phys_health],
        "MentalHealth": [ment_health],
        "SleepTime": [sleep_time],
        "BMICategory": [bmi_cat],
        "Smoking": [smoking],
        "AlcoholDrinking": [alcohol_drink],
        "DiffWalking": [diff_walk],
        "Sex": [sex],
        "AgeCategory": [age_cat],
        "Diabetic": [diabetic],
        "PhysicalActivity": [phys_act],
        "GenHealth": [gen_health],
        "Asthma": [asthma],
        "KidneyDisease": [kid_dis],
        "SkinCancer": [skin_canc]
    })

    return features

# Set page configuration outside of any function
st.set_page_config(
    page_title="ByteForza - Heart Attack Prediction App",
    page_icon="images/heart-fav.png"
)

# The main function
def main():
    # Load dataset
    heart = load_dataset()

    # Set up the title and description
    st.title("Heart Attack Prediction")
    st.subheader("Concerned about your heart health? This app is here to help you assess your risk and take proactive steps to safeguard your heart!")

    # Display the image and prediction button
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("images/aidocter.jpg",
                 caption="I'll help you diagnose your heart health! - Dr. ByteForza AI",
                 width=200)
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

    # Sidebar for user inputs
    st.sidebar.title("Feature Selection")
    st.sidebar.image("images/heart-sidebar.png", width=100)
    
    # Get user input features
    input_df = user_input_features(heart)
    
    if submit:
        st.write("User input features:")
        st.write(input_df)

if __name__ == "__main__":
    main()
