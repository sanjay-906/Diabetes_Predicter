# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:33:49 2022

@author: Admin
"""

import numpy as np
import pickle
import streamlit as sl

pickle_in=open("logreg.sav","rb")
model2=pickle.load(pickle_in)

def compute(input_values):
    
    pred_array=np.asarray(input_values).reshape(1,-1)
    predictions=model2.predict(pred_array)
    if(predictions[0]==1):
        return "High chance of getting diabetes"
    else:
        return "Low chance of getting diabetes"
    
    
    
def main():
    
    sl.title('Diabetes Checker')
    
    Pregnancies = sl.number_input("Number of Pregnancies: ")
    Glucose = sl.number_input("Glucose level: ")
    BloodPressure = sl.number_input("Blood Pressure: ")
    SkinThickness = sl.number_input("Skin Thickness: ")
    Insulin = sl.number_input("Insulin level: ")
    BMI = sl.number_input("Body Mass Index: ")
    DiabetesPedigreeFunction = sl.number_input("DiabetesPedigreeFunction: ")
    Age = sl.number_input("Age: ")
    
    
    
    patient=''
    if sl.button("Evaluate"):
        patient= compute([Pregnancies,Glucose, BloodPressure, 
                 SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    sl.success(patient)
 
 
 
if __name__ == '__main__':
    main()
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
