import streamlit as st
import pandas as pd
import joblib

# load model and encoder
model = joblib.load("salary_prediction_rfr_model (1).pkl")
encoder = joblib.load("encoder.pkl")

st.title("Salary Prediction Model")

age = st.number_input("Enter your age", 18, 65)

gender = st.selectbox("Select your gender", encoder["Gender"].classes_)
education = st.selectbox("Select your education", encoder["Education Level"].classes_)
job_title = st.selectbox("Select your job title", encoder["Job Title"].classes_)

experience = st.number_input("Enter your experience (in years)", 0, 50)

# create dataframe from user input
df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education Level": [education],
    "Job Title": [job_title],
    "Years of Experience": [experience]
})

# prediction button
if st.button("Predict Salary"):

    for col in encoder:
        df[col] = encoder[col].transform(df[col])

    prediction = model.predict(df)

    st.success(f"Predicted Salary: {prediction[0]}")
