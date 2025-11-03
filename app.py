#Gender-->1 Female else 0 Male
#Churn-->1 Yes else 0 No
#scaler is exported as scaler.pkl
#model is exported as model.pkl
#order of  x columns is Age', 'Gender', 'Tenure', 'MonthlyCharges'], dtype='object'


import streamlit as st
import joblib
import numpy as np

scaler=joblib.load("scaler.pkl")
model=joblib.load("model.pkl")


st.title("Customer Churn Prediction App")
st.divider()
st.write("This app predicts whether a customer will churn or not based on their details.")
st.divider()
age=st.number_input("Enter Age",min_value=10,max_value=130 ,value=25)
tenure=st.number_input("Enter Tenure",min_value=0,max_value=72,value=12)
monthly_charges=st.number_input("Enter Monthly Charges",min_value=0.0,max_value=10000.0,value=70.0)
gender_str=st.selectbox("Select Gender",["Male","Female"])

st.divider()
predict_button=st.button("Predict Churn")
st.divider()
if predict_button:
    gender_numeric = 1 if gender_str == "Female" else 0
    input_features = np.array([[age, gender_numeric, tenure, monthly_charges]])
    scaled_features = scaler.transform(input_features)
    prediction = model.predict(scaled_features)[0]
    predicted="Yes"if prediction==1 else "No"
    st.balloons()
    st.success(f"Predicted Churn: {predicted}")
else:
    st.write("Click on Predict Churn button to get the prediction")