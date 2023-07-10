import joblib 
import pandas as pd
import numpy as np
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import gradio as gr
import joblib
import warnings


warnings.filterwarnings("ignore")

best_model= joblib.load("models/RF.joblib")

best_model

test_data= pd.read_csv("dataframes\Telco_churn.csv")
test_data

##test the model
best_model.predict(test_data)

##create a function to return a string depending on the output of the model

def classify(num):
    if num == 0:
        return "Customer will not Churn"
    else:
        return "Customer will churn"


"""create a function for my gradio fn
define my parameters which my fucntion will accept, and are the same as the features I trained my model on"""


def predict_churn(SeniorCitizen, Partner, Dependents, tenure, InternetService,
                  OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                  StreamingTV, StreamingMovies, Contract, PaperlessBilling,
                  PaymentMethod, MonthlyCharges, TotalCharges): 

     
     ## created a list of my input features

    input_data = [
        SeniorCitizen, Partner, Dependents, tenure, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract, PaperlessBilling,
        PaymentMethod, MonthlyCharges, TotalCharges
    ]    
## changing my features into a dataframe since that is how I trained my model

    input_df = pd.DataFrame([input_data], columns=[
        "SeniorCitizen", "Partner", "Dependents", "tenure", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ])


    pred = best_model.predict(input_df) ## make a prediction on the input data.

    output = classify(pred[0]) ## pass the first predction through my classify function I created earlier

    if output == "Customer will not Churn":
        return [(0, output)]
    else:
        return [(1, output)]   ##setting my function to return the binary classification and the written output

output = gr.outputs.HighlightedText(color_map={
    "Customer will not Churn": "green",
    "Customer will churn": "red"
}) ##assign colors to the respective output 

##building my interface and wrapping my model in the function

## using gradio blocks to beautify my output

block= gr.Blocks() ##instantiating my blocks class

with block:
    gr.Markdown(""" # Customer Churn Prediction App""")
    
    input=[gr.inputs.Slider(minimum=0, maximum= 1, step=1, label="SeniorCitizen: Select 1 for Yes and 0 for No"),
        gr.inputs.Radio(["Yes", "No"], label="Partner: Do You Have a Partner?"),
        gr.inputs.Radio(["Yes", "No"], label="Dependents: Do You Have a Dependent?"),
        gr.inputs.Number(label="tenure: How Long Have You Been with Vodafone in Months?"),
        gr.inputs.Radio(["DSL", "Fiber optic", "No"], label="InternetService"),
        gr.inputs.Radio(["Yes", "No", "No internet service"], label="OnlineSecurity"),
        gr.inputs.Radio(["Yes", "No", "No internet service"], label="OnlineBackup"),
        gr.inputs.Radio(["Yes", "No", "No internet service"], label="DeviceProtection"),
        gr.inputs.Radio(["Yes", "No", "No internet service"], label="TechSupport"),
        gr.inputs.Radio(["Yes", "No", "No internet service"], label="StreamingTV"),
        gr.inputs.Radio(["Yes", "No", "No internet service"], label="StreamingMovies"),
        gr.inputs.Radio(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.inputs.Radio(["Yes", "No"], label="PaperlessBilling"),
        gr.inputs.Radio([
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], label="PaymentMethod"),
        gr.inputs.Number(label="MonthlyCharges"),
        gr.inputs.Number(label="TotalCharges")]
     
    output= gr.outputs.HighlightedText(color_map={
     "Customer will not Churn": "purple",
     "Customer will churn": "pink"}, label= "Your Output")     
    predict_btn= gr.Button("Predict")
      # Expander for more info on columns
    with gr.Accordion("Open for information on inputs"):
        gr.Markdown("""This app receives the following as inputs and processes them to return the prediction on whether a customer, given the inputs, will churn or not.
                    - Contract: The contract term of the customer (Month-to-Month, One year, Two year)
                    - Dependents: Whether the customer has dependents or not (Yes, No)
                    - DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)
                    - Gender: Whether the customer is a male or a female
                    - InternetService: Customer's internet service provider (DSL, Fiber Optic, No)
                    - MonthlyCharges: The amount charged to the customer monthly
                    - MultipleLines: Whether the customer has multiple lines or not
                    - OnlineBackup: Whether the customer has online backup or not (Yes, No, No Internet)
                    - OnlineSecurity: Whether the customer has online security or not (Yes, No, No Internet)
                    - PaperlessBilling: Whether the customer has paperless billing or not (Yes, No)
                    - Partner: Whether the customer has a partner or not (Yes, No)
                    - Payment Method: The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))
                    - Phone Service: Whether the customer has a phone service or not (Yes, No)
                    - SeniorCitizen: Whether a customer is a senior citizen or not
                    - StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No Internet service)
                    - StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service)
                    - TechSupport: Whether the customer has tech support or not (Yes, No, No internet)
                    - Tenure: Number of months the customer has stayed with the company
                    - TotalCharges: The total amount charged to the customer
                    """)
    predict_btn.click(fn= predict_churn, inputs= input, outputs=output)

block.launch(favicon_path=r"src\app_thumbnail.png",
                          inbrowser= True)


