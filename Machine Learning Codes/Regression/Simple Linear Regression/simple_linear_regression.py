#Simple Linear reression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting simple regression model in test data set
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)


#predicting the test set results
y_pred = regression.predict(X_test)

#visualize the training set results
plt.scatter(X_train,y_train,color = 'Red')
plt.plot(X_train, regression.predict(X_train), color = 'Blue')
plt.title('Salary vs Experience for Training set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualize the test set results
plt.scatter(X_test,y_test,color = 'Red')
plt.plot(X_train, regression.predict(X_train), color = 'Blue')
plt.title('Salary vs Experience for Test set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
