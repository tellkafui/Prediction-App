import streamlit as st
import numpy as np
import pandas as pd
import joblib 
from datetime import date

##setting my main page
st.set_page_config(page_title="Time Series Forecast App"
                   )

## title
st.title("Favorita Store Sales Prediction App")
##adding my description 
st.markdown("Make your daily sales prediction across Favorita Stores")

##loading model

model= joblib.load("model/fbpmodel.joblib")

##loading my test data

test=pd.read_csv("data/test.csv")

test=test.drop(["holiday", "locale", "transferred"], axis= 1)



##defining my inputs 
st.header("Make your predictions Here: ")
ds= st.date_input(label= "Select the date you would want to forecast")
transactions= st.number_input(label= "Enter the total number of expected transactions")
onpromotion= st.number_input(label= "Enter the total number of expected items to be on promotions")

##creating a dataframe for my inputs 

input_data= [ds, onpromotion, transactions]
inputs= pd.DataFrame([input_data], columns=["ds", "onpromotion", "transactions"])
forecast= model.predict(inputs)
forecast_value= forecast["yhat"]
forecast_output = f" Total sales on {ds} will be ${forecast_value.values[0]:.2f}"

##creating an empty dataframe which will be displayed until the user clicks on submit
forecast_emp = forecast.applymap(lambda x: np.nan)
forecast_emp= forecast_emp.replace(np.nan,0) 

## button
output= st.button("Submit")

st.subheader("Your Sales Forecast is Displayed Below: ")

##telling my model to return the yhat if the submit button is clicked
if output:
    st.write(forecast_output)
else:
    st.write("Total Sales On ___ Will Be _______")
