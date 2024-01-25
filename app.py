"""
Created on Sun Jan 21 19:36:12 2024

@author: Swapnil
"""

import numpy as np
import pandas as pd
#import pickle 
import streamlit as st
from PIL import Image
from pycaret.classification import load_model, predict_model
#pickle_in=open('C:/Users/Swapnil/Desktop/Unified mentor/Employee attrition/Employee-Attrition-Analysis/best-model.pkl',"rb")
#classifier=pickle.load(pickle_in)


def predict_rf(MonthlyIncome,TotalWorkingYears,Age,DistanceFromHome,PercentSalaryHike):
    dataa={
        'EmployeeID':[1],
        'Age':[Age],
        'BusinessTravel':['Travel_Rarely'],
        'Department':['Sales'],
        'DistanceFromHome':[DistanceFromHome],
        'Education':[2],
        'EducationField':['Life Sciences'],
        'EmployeeCount':[1],
        'Gender':['Male'],
        'JobLevel':[1],
        'JobRole':['Healthcare Representative'],
        'MaritalStatus':['Single'],
        'MonthlyIncome':[MonthlyIncome],
        'NumCompaniesWorked':[1],
        'Over18':['Y'],
        'PercentSalaryHike':[PercentSalaryHike],
        'StandardHours':[8],
        'StockOptionLevel':[1],
        'TotalWorkingYears':[TotalWorkingYears],
        'TrainingTimesLastYear':[6],
        'YearsAtCompany':[1],
        'YearsSinceLastPromotion':[0],
        'YearsWithCurrManager':[0],
        'EnvironmentSatisfaction':[3],
        'JobSatisfaction':[4],
        'WorkLifeBalance':[2],
        'JobInvolvement':[3],
        'PerformanceRating':[3]
        }
    
    df=pd.DataFrame(dataa)
    tuned_rf=load_model('C:/Users/Swapnil/Desktop/Unified mentor/Employee attrition/Employee-Attrition-Analysis/best-model')
    prediction=predict_model(tuned_rf,data=pd.DataFrame(dataa))
    
    return prediction['prediction_label'][0]

def main():
    st.title("Employee Attrition Predictor")
    html_temp="""
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Please enter the following employee details </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    MonthlyIncome=st.text_input("Monthly Income")
    TotalWorkingYears=st.text_input("Total Working Years")
    Age=st.text_input("Age")
    DistanceFromHome=st.text_input("Distance From Home")
    PercentSalaryHike=st.text_input("Percent Salary Hike")
    result=""
    if st.button("Predict"):
        result=predict_rf(MonthlyIncome, TotalWorkingYears, Age, DistanceFromHome, PercentSalaryHike)
    st.success('Employee Attrition Prediction is: {}'.format(result))
    if st.button("About"):
        st.text("Created by Swapnil Gavali")

if __name__=='__main__':
    main()