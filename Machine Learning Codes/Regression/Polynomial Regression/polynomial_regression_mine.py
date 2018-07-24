# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 22:17:34 2018

@author: Gaurav
"""

#Polynomial_Linear_Regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""



#fitting the X dataset
from sklearn.linear_model import LinearRegression
Lin_reg = LinearRegression()
Lin_reg.fit(X,y)


#fitting the polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
Poly_reg = PolynomialFeatures(degree=7)
X_poly = Poly_reg.fit_transform(X)
from sklearn.linear_model import LinearRegression
Lin_reg2 = LinearRegression()
Lin_reg2.fit(X_poly,y)


#Visualizing the Linear regression model
plt.scatter(X,y,color='Red')
plt.plot(X, Lin_reg.predict(X) , color = 'Blue')
plt.title('Truth or Bluff Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial regression model
X_grib = np.arange(min(X),max(X),0.1)
X_grib = X_grib.reshape((len(X_grib),1))
plt.scatter(X,y,color='Red')
plt.plot(X_grib, Lin_reg2.predict(Poly_reg.fit_transform(X_grib)) , color = 'Blue')
plt.title('Truth or Bluff Polynomial LInear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#predicting the salary with linaer regression
Lin_reg.predict(6.5)



#predictiong the salary with polinomial regression
Lin_reg2.predict(Poly_reg.fit_transform(6.5))




