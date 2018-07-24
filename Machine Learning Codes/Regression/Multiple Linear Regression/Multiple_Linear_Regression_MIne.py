# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 01:58:40 2018

@author: Gaurav
"""

#Multiple_regresssion_Model


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values





# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding_the_Dummy_Variable_Trap
X=X[:,1:]




# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting the Multiple Linear REgression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#predicting the test results
y_pred = regressor.predict(X_test)

#Building an Backward Elimanation model
import statsmodels.formula.api as sm
X=np.append(arr = np.ones((50,1)).astype(int) ,values = X ,axis = 1)
X_opt =  X[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y , exog =  X_opt).fit()
regressor_ols.summary()
X_opt =  X[:,[0,1,2,4,5]]
regressor_ols = sm.OLS(endog = y , exog =  X_opt).fit()
regressor_ols.summary()
X_opt =  X[:,[0,1,4,5]]
regressor_ols = sm.OLS(endog = y , exog =  X_opt).fit()
regressor_ols.summary()
X_opt =  X[:,[0,3]]
regressor_ols = sm.OLS(endog = y , exog =  X_opt).fit()
regressor_ols.summary()
