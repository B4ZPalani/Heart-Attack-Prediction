import streamlit as st
import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import joblib
from sklearn.linear_model import LogisticRegression


@st.cache_resource
def loadModels():
        # Load the models dictionary
    filename = 'model/heart-attack-models.pkl'
    return joblib.load(open(filename, 'rb'))

st.subheader("Heart Attack Prediction1")
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("""
    To predict your heart disease status, simply follow the steps below:
    1. Enter the parameters that best describe you;
    2. Press the "Predict" button and wait for the result.
    """)
with col2:
    st.image("images/heart-sidebar.png", width=100)

submit = st.button("Predict")


# st.write(BMIdata)

# Thesidebar func tion from streamlit is used to create a sidebar for users 
# to input their information.
# -------------------------------------------------------------------------
st.sidebar.title('Please, fill your informations to predict your heart condition')


Race=st.sidebar.selectbox("Select your Race", ("Asian", 
                             "Black" ,
                             "Hispanic",
                             "American Indian/Alaskan Native",
                             "White",
                             "Other"
                             ))
Gender=st.sidebar.selectbox("Select your gender", ("Female", 
                             "Male" ))
Age=st.sidebar.selectbox("Select your age", 
                            ("18-24", 
                             "25-29" ,
                             "30-34",
                             "35-39",
                             "40-44",
                             "45-49",
                             "50-54",
                             "55-59",
                             "60-64",
                             "65-69",
                             "70-74",
                             "75-79",
                             "55-59",
                             "80 or older"))

BMI=st.sidebar.number_input("BMI",18,100,18)

Smoking = st.sidebar.selectbox("Have you smoked more than 100 cigarettes in"
                          " your entire life ?)",
                          options=("No", "Yes"))
alcoholDink = st.sidebar.selectbox("How many drinks of alcohol do you have in a week?", options=("No", "Yes"))

sleepTime = st.sidebar.number_input("Hours of sleep per 24h", 0, 24, 7) 

genHealth = st.sidebar.selectbox("General health",
                             options=("Good","Excellent", "Fair", "Very good", "Poor"))

physHealth = st.sidebar.number_input("Physical health in the past month (Excelent: 0 - Very bad: 30)"
                                 , 0, 30, 0)
mentHealth = st.sidebar.number_input("Mental health in the past month (Excelent: 0 - Very bad: 30)"
                                 , 0, 30, 0)
physAct = st.sidebar.selectbox("Physical activity in the past month"
                           , options=("No", "Yes"))

diffWalk = st.sidebar.selectbox("Do you have serious difficulty walking"
                            " or climbing stairs?", options=("No", "Yes"))
diabetic = st.sidebar.selectbox("Have you ever had diabetes?",
                           options=("No", "Yes", "Yes, during pregnancy", "No, borderline diabetes"))
asthma = st.sidebar.selectbox("Do you have asthma?", options=("No", "Yes"))
kidneyDisease= st.sidebar.selectbox("Do you have kidney disease?", options=("No", "Yes"))
skinCancer = st.sidebar.selectbox("Do you have skin cancer?", options=("No", "Yes"))

dataToPredic = pd.DataFrame({
   "BMI": [BMI],
   "Smoking": [Smoking],
   "AlcoholDrinking": [alcoholDink],
   "Stroke":["No"],
   "PhysicalHealth": [physHealth],
   "MentalHealth": [mentHealth],
   "DiffWalking": [diffWalk],
   "Sex": [Gender],
   "AgeCategory": [Age],
   "Race": [Race],
   "Diabetic": [diabetic],
   "PhysicalActivity": [physAct],
   "GenHealth": [genHealth],
   "SleepTime": [sleepTime],
   "Asthma": [asthma],
   "KidneyDisease": [kidneyDisease],
   "SkinCancer": [skinCancer]
 })


dataToPredic.replace("Yes",1,inplace=True)
dataToPredic.replace("No",0,inplace=True)
dataToPredic.replace("18-24",0,inplace=True)
dataToPredic.replace("25-29",1,inplace=True)
dataToPredic.replace("30-34",2,inplace=True)
dataToPredic.replace("35-39",3,inplace=True)
dataToPredic.replace("40-44",4,inplace=True)
dataToPredic.replace("45-49",5,inplace=True)
dataToPredic.replace("50-54",6,inplace=True)
dataToPredic.replace("55-59",7,inplace=True)
dataToPredic.replace("60-64",8,inplace=True)
dataToPredic.replace("65-69",9,inplace=True)
dataToPredic.replace("70-74",10,inplace=True)
dataToPredic.replace("75-79",11,inplace=True)
dataToPredic.replace("80 or older",13,inplace=True)


dataToPredic.replace("No, borderline diabetes",2,inplace=True)
dataToPredic.replace("Yes (during pregnancy)",3,inplace=True)


dataToPredic.replace("Excellent",0,inplace=True)
dataToPredic.replace("Good",1,inplace=True)
dataToPredic.replace("Fair",2,inplace=True)
dataToPredic.replace("Very good",3,inplace=True)
dataToPredic.replace("Poor",4,inplace=True)


dataToPredic.replace("White",0,inplace=True)
dataToPredic.replace("Other",1,inplace=True)
dataToPredic.replace("Black",2,inplace=True)
dataToPredic.replace("Hispanic",3,inplace=True)
dataToPredic.replace("Asian",4,inplace=True)
dataToPredic.replace("American Indian/Alaskan Native",4,inplace=True)


dataToPredic.replace("Female",0,inplace=True)
dataToPredic.replace("Male",1,inplace=True)

@st.cache_resource
def loadModels():
        # Load the models dictionary
    filename = 'model/heart-attack-models.pkl'
    return joblib.load(open(filename, 'rb'))

if submit:

    loaded_models = loadModels()
    # Access each model
    logistic_model = loaded_models['LogisticRegression']
    random_forest_model = loaded_models['RandomForest']

    # Predict using both models
    logistic_result = logistic_model.predict(dataToPredic)
    logistic_result_prob = logistic_model.predict_proba(dataToPredic)
    logistic_result_prob1 = round(logistic_result_prob[0][1] * 100, 2)

    random_forest_result = random_forest_model.predict(dataToPredic)
    random_forest_result_prob = random_forest_model.predict_proba(dataToPredic)
    random_forest_result_prob1 = round(random_forest_result_prob[0][1] * 100, 2)
    
    
    # loaded_model= pickle.load(open('LogRegModel.pkl', 'rb'))
    # Result=loaded_model.predict(dataToPredic)
    # ResultProb= loaded_model.predict_proba(dataToPredic)
    # ResultProb1=round(ResultProb[0][1] * 100, 2)


    # if (ResultProb1>30):
    #     st.write('You have a', ResultProb1, '% chance of getting a heart disease' )
    # else:
    #     st.write('You have a', ResultProb1, '% chance of getting a heart disease' )
    
    
    
    
    
    

    st.write("Your heart disease predictions are ready!")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Log. Regression Prediction")
        if logistic_result_prob1 < 30:
            st.markdown(f"**The probability that you'll have heart disease is {round(logistic_result_prob[0][1] * 100, 2)}%. You are healthy!**")
            st.image("images/heart-okay.jpg", caption="Your heart seems to be okay! - Dr. ByteForza Regression")
        else:
            st.markdown(f"**The probability that you'll have heart disease is {round(logistic_result_prob[0][1] * 100, 2)}%. You might be at risk!**")
            st.image("images/heart-bad.jpg", caption="Your heart might be at risk! - Dr. ByteForza Regression")
    with col2:
        st.subheader("Random Forest Prediction")
        if random_forest_result_prob1 < 30:
            st.markdown(f"**The probability that you'll have heart disease is {round(random_forest_result_prob[0][1] * 100, 2)}%. You are healthy!**")
            st.image("images/heart-okay.jpg", caption="Your heart seems to be okay! - Dr. ByteForza Forest")
        else:
            st.markdown(f"**The probability that you'll have heart disease is {round(random_forest_result_prob[0][1] * 100, 2)}%. You might be at risk!**")
            st.image("images/heart-bad.jpg", caption="Your heart might be at risk! - Dr. ByteForza Forest")

  

  
