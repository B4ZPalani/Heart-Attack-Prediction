import streamlit as st
import pandas as pd
import numpy as np
import sklearn

# Set page configuration outside of any function
st.set_page_config(
    page_title="ByteForza - Heart Attack Prediction App",
    page_icon="images/heart-fav.png"
)

dataSetPath = "data/heart_2020_cleaned.csv"

def createUserInput(heart: pd.DataFrame) -> pd.DataFrame:
    race = st.sidebar.selectbox("Race", options=heart['Race'].unique(), key="Race")
    sex = st.sidebar.selectbox("Sex", options=heart['Sex'].unique(), key="sex")
    age_cat = st.sidebar.selectbox("Age category", options=heart['AgeCategory'].unique(), key="age_cat")
    bmi_cat = st.sidebar.selectbox("BMI category", options=heart['BMICategory'].unique(), key="bmi_cat")
    sleep_time = st.sidebar.number_input("How many hours on average do you sleep?", 0, 24, 7, key="sleep_time")
    gen_health = st.sidebar.selectbox("How can you define your general health?", options=heart['GenHealth'].unique(), key="gen_health")
    phys_health = st.sidebar.number_input("For how many days during the past 30 days was your physical health not good?", 0, 30, 0, key="phys_health")
    ment_health = st.sidebar.number_input("For how many days during the past 30 days was your mental health not good?", 0, 30, 0, key="ment_health")
    phys_act = st.sidebar.selectbox("Have you played any sports (running, biking, etc.) in the past month?", options=("No", "Yes"), key="phys_act")
    smoking = st.sidebar.selectbox("Have you smoked at least 100 cigarettes in your entire life (approx. 5 packs)?", options=("No", "Yes"), key="smoking")
    alcohol_drink = st.sidebar.selectbox("Do you have more than 14 drinks of alcohol (men) or more than 7 (women) in a week?", options=("No", "Yes"), key="alcohol_drink")
    diff_walk = st.sidebar.selectbox("Do you have serious difficulty walking or climbing stairs?", options=("No", "Yes"), key="diff_walk")
    diabetic = st.sidebar.selectbox("Have you ever had diabetes?", options=heart['Diabetic'].unique(), key="diabetic")
    asthma = st.sidebar.selectbox("Do you have asthma?", options=("No", "Yes"), key="asthma")
    kid_dis = st.sidebar.selectbox("Do you have kidney disease?", options=("No", "Yes"), key="kid_dis")
    skin_canc = st.sidebar.selectbox("Do you have skin cancer?", options=("No", "Yes"), key="skin_canc")
    
    features = pd.DataFrame({
        "BMICategory": [bmi_cat],
        "Smoking": [smoking],
        "AlcoholDrinking": [alcohol_drink],
        "PhysicalHealth": [phys_health],
        "MentalHealth": [ment_health],
        "DiffWalking": [diff_walk],
        "Sex": [sex],
        "AgeCategory": [age_cat],
        "Race":[race],
        "Diabetic": [diabetic],
        "PhysicalActivity": [phys_act],
        "GenHealth": [gen_health],
        "SleepTime": [sleep_time],
        "Asthma": [asthma],
        "KidneyDisease": [kid_dis],
        "SkinCancer": [skin_canc]
    })
    return features

@st.cache_data
def readDataset():
    df = pd.read_csv(dataSetPath)
    return df

def basicCleansing(df):
    #st.write("before cleansing")
    #st.write(df.head(100))
    df['Smoking'] = pd.Series(np.where(df['Smoking'] == 'Yes', 1, 0))
    df['AlcoholDrinking'] = pd.Series(np.where(df['AlcoholDrinking'] == 'Yes', 1, 0))
    df['DiffWalking'] = pd.Series(np.where(df['DiffWalking'] == 'Yes', 1, 0))
    df['PhysicalActivity'] = pd.Series(np.where(df['PhysicalActivity'] == 'Yes', 1, 0))
    df['Asthma'] = pd.Series(np.where(df['Asthma'] == 'Yes', 1, 0))
    df['KidneyDisease'] = pd.Series(np.where(df['KidneyDisease'] == 'Yes', 1, 0))
    df['SkinCancer'] = pd.Series(np.where(df['SkinCancer'] == 'Yes', 1, 0))
    if 'HeartDisease' in df.columns:
        df['HeartDisease'] = pd.Series(np.where(df['HeartDisease'] == 'Yes', 1, 0))

    # Set up the Label Encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['AgeCategory'] = le.fit_transform(df['AgeCategory'])
    df['BMICategory'] = le.fit_transform(df['BMICategory'])
    df['Race'] = le.fit_transform(df['Race'])
    df['Diabetic'] = le.fit_transform(df['Diabetic'])
    df['GenHealth'] = le.fit_transform(df['GenHealth'])

    if 'Stroke' in df.columns:
        df.drop(['Stroke'],axis=1,inplace=True)
    if 'PhysicalActivity' in df.columns:
        df.drop(['PhysicalActivity'],axis=1,inplace=True)
    if 'GenHealth' in df.columns:
        df.drop(['GenHealth'], axis=1,inplace=True)
    if 'SleepTime' in df.columns:
        df.drop(['SleepTime'], axis=1,inplace=True)        
    return df

def splitTestTrainData(df):
    df = basicCleansing(df)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.2, random_state=42)


def trainLogisticRegressionModel(X_train, y_train,X_test, y_test):
    # Initialize and train the Logistic Regression model
    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    #st.write(X_train.head(100))
    lr_model.fit(X_train, y_train)
    y_lr_pred = lr_model.predict(X_test)
    st.write("LogisticRegression")
    st.write(y_lr_pred)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_lr_pred)
    st.write("LogisticRegression Accuracy Score")
    st.write(f'Accuracy: {accuracy:.2f}')

    # unique, counts = np.unique(y_lr_pred, return_counts=True)
    # count_dict = dict(zip(unique, counts))
    # st.write(count_dict)
    return lr_model

def trainRandomForestModel(X_train, y_train,X_test, y_test):
    # Initialize and train the Logistic Regression model
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    #st.write(X_train.head(100))
    rf_model.fit(X_train, y_train)
    y_rf_pred = rf_model.predict(X_test)
    st.write("RandomForestClassifier")
    st.write(y_rf_pred)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_rf_pred)
    st.write("RandomForestClassifier Accuracy Score")
    st.write(f'Accuracy: {accuracy:.2f}')
    # unique, counts = np.unique(y_lr_pred, return_counts=True)
    # count_dict = dict(zip(unique, counts))
    # st.write(count_dict)
    return rf_model

def createUIElements():
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
    heart_data = readDataset()
    user_input = createUserInput(heart_data)

    if submit:
        predictionByUserInput(user_input)

def predictionByUserInput(user_input):
    df = readDataset()
    X_train, X_test, y_train, y_test = splitTestTrainData(df)
    cleaned_df = basicCleansing(user_input)


    lr_model = trainLogisticRegressionModel(X_train, y_train,X_test, y_test)
    lr_prediction = lr_model.predict(cleaned_df)
    Lr_prediction_prob = lr_model.predict_proba(cleaned_df)

    rf_model = trainRandomForestModel(X_train, y_train,X_test, y_test)
    rf_prediction = rf_model.predict(cleaned_df)
    fr_prediction_prob = rf_model.predict_proba(cleaned_df)

    st.subheader("Logistic Regression Prediction")
    if lr_prediction == 0:
            st.markdown(f"**The probability that you'll have"
                        f" heart disease is {round(Lr_prediction_prob[0][1] * 100, 2)}%."
                        f" You are healthy!**")
            st.image("images/heart-okay.jpg",
                     caption="Your heart seems to be okay! - Dr. ByteForza Regression")
    else:
            st.markdown(f"**The probability that you will have"
                        f" heart disease is {round(Lr_prediction_prob[0][1] * 100, 2)}%."
                        f" It sounds like you are not healthy.**")
            st.image("images/heart-bad.jpg",
                     caption="I'm not satisfied with the condition of your heart! - Dr. ByteForza Regression")
            
    st.subheader("Random Forest Prediction")
    if rf_prediction == 0:
            st.markdown(f"**The probability that you'll have"
                        f" heart disease is {round(fr_prediction_prob[0][1] * 100, 2)}%."
                        f" You are healthy!**")
            st.image("images/heart-okay.jpg",
                     caption="Your heart seems to be okay! - Dr. ByteForza Forest")
    else:
            st.markdown(f"**The probability that you will have"
                        f" heart disease is {round(fr_prediction_prob[0][1] * 100, 2)}%."
                        f" It sounds like you are not healthy.**")
            st.image("images/heart-bad.jpg",
                     caption="I'm not satisfied with the condition of your heart! - Dr. ByteForza Forest")

def main():
    createUIElements()

if __name__ == "__main__":
    main()
