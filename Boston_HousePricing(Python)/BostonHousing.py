# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:09:27 2019

@author: hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("E:\\Shanu\\Machine-Learning-Projects\\Boston_HousePricing(Python)\\BostonHousing.csv", delim_whitespace= True, header= None)
col_names = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]

df.columns = col_names

pd.options.display.float_format = '{:,.2f}'.format

df.corr()
df.head()

plt.figure(figsize=(16,10))
sns.heatmap(df.corr(),annot= True)
plt.show()

df.isna().sum()

sns.heatmap(df[["CRIM","ZN","INDUS","AGE","LSTAT"]].corr(),annot= True)
plt.show()

##Look for the correlation between vraiables . do not choose variables having high positive or negative correlation. DO not taken those variabes
# together to train the model

plt.scatter(df['LSTAT'],df['MEDV']) 
plt.scatter(df['RM'],df['MEDV']) 

df.shape
X= pd.DataFrame(np.c_[df['LSTAT'],df['RM']]).values
y = df['MEDV'].values


## Splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.2,random_state=1)

# Appying Linear Regression to the dataset

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred =regressor.predict(X_test)


### Evaluate the model

y_train_predict = regressor.predict(X_train)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


rmse_train = np.sqrt(mean_squared_error(y_train,y_train_predict))
r2_train = r2_score(y_train, y_train_predict)

print("Root Mean Squared error for training set is: ", rmse_train)
print("r squared value for training set is: ", r2_train)

rmse_test = np.sqrt(mean_squared_error(y_test,y_pred))
r2_test = r2_score(y_test,y_pred)

print("Root Mean Squared error for test set is: ", rmse_test)
print("r squared value for test set is: ", r2_test)




