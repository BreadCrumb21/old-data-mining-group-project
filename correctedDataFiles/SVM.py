# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 

data = pd.read_csv('hotData.csv')
print(data.isna().sum())
X = data.drop(columns=['Workclass_new', 'Education_new', 'Marital-status_new', 'Occupation_new', 'Relationship_new', 'Race_new', 'Sex_new','Native-country_new', 'Salary_new'])
y = data['Salary_new']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size= 0.30, random_state= 2)
moddel = SVC().fit(X_train, Y_train)
print(moddel.score(X_test, Y_test))