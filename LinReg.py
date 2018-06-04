# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:05:05 2018

@author: adam
"""
import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.cross_validation
import matplotlib.pyplot as plt 

def linear_regression(X_learn,X_test,Y_learn,Y_test,X,Y):
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_learn,Y_learn)
    Y_learn_pred = model.predict(X_learn)
    Y_test_pred = model.predict(X_test)
    Y_pred = model.predict(X)
    mse = sklearn.metrics.mean_squared_error
    return{
            "Coefficient of determination": model.score(X_learn,Y_learn),
            "Mean squared error - learn set": mse(Y_learn,Y_learn_pred),
            "Mean squared error - test set": mse(Y_test,Y_test_pred),
            }

M=pd.read_csv("buildondata_dataset_interview.csv")
A=M.iloc[:,2:]
lista=[]
lista = [i for i in A.columns if i!="ACTIVE POWER" and i!="REACTIVE POWER"]
X=A.loc[:,lista]
Y_AP=A["ACTIVE POWER"]
Y_RP=A["REACTIVE POWER"]
X_learn,X_test,Y_learn,Y_test = sklearn.cross_validation.train_test_split(X,Y_AP,test_size=0.2,random_state=12345)
result=linear_regression(X_learn,X_test,Y_learn,Y_test,X,Y_AP)
print("Results for active power: ",result)
X_learn,X_test,Y_learn,Y_test = sklearn.cross_validation.train_test_split(X,Y_RP,test_size=0.2,random_state=12345)
result=linear_regression(X_learn,X_test,Y_learn,Y_test,X,Y_RP)
print("Results for reactive power: ",result)
