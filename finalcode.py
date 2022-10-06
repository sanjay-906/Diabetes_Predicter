# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:35:14 2022

@author: sanjay_906_
"""
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import streamlit as sl  

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv('diabetes.csv')
data=shuffle(data)
X=data.drop(columns='Outcome',axis=1)
Y=data['Outcome']


scaler=StandardScaler()
new_data=scaler.fit_transform(X)
X=new_data
y=data['Outcome']


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,stratify=y)
model=LogisticRegression(penalty="l2", C=10)
model.fit(X_train,y_train)



X_train_pred=model.predict(X_train)
acc_train=accuracy_score(X_train_pred,y_train)

X_test_pred=model.predict(X_test)
acc_test=accuracy_score(X_test_pred,y_test)
print("Accuracy on Training Data:",acc_train*100,"%")
print("Accuracy on Test Data:",acc_test*100,"%")


def compute(input_values):
    
    pred_array=np.asarray(input_values).reshape(1,-1)
    predictions=model.predict(pred_array)
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
