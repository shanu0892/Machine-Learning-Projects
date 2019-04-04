import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("C:\\Users\\Arun\\Documents\\shanu\\kaggle\\BlackFriday.csv")
df.describe()
df.info()

df= df.drop(["Product_ID","User_ID"], axis = 1)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values= "NaN", strategy = "mean", axis =0)
imputer = imputer.fit(df.iloc[:,7:9])
df.iloc[:,7:9] = imputer.transform(df.iloc[:,7:9].values)

df["Age"] = df["Age"].str.strip('+')
df["Stay_In_Current_City_Years"] = df["Stay_In_Current_City_Years"].str.strip('+')

sns.heatmap(df.corr(),annot= True)

g = sns.FacetGrid(df,col = "Age")
g.map(sns.barplot,"Gender","Purchase")

g = sns.FacetGrid(df,col = "Stay_In_Current_City_Years")
g.map(sns.barplot,"Marital_Status","Purchase")
# the above plot is difficult to conclude

g = sns.FacetGrid(df,col = "Marital_Status")    
g.map(sns.barplot,"Gender","Purchase")
# Male purchase more than Female after marriage

g = sns.FacetGrid(df,row = "Age", col = "Gender")
g.map(sns.barplot,"Marital_Status","Purchase")

g = sns.FacetGrid(df,col = "City_Category")    
g.map(sns.barplot,"Gender","Purchase")
#Category B and C have higher purchasing in males

sns.jointplot(x='Occupation',y='Purchase',
              data=df, kind='hex')
#the above plot shows that adverstisement be given to those customers whose purchase value is between 5000 and 10000

g = sns.FacetGrid(df)
g.map(sns.barplot,"Gender","Purchase")
# male purchase more than Female

df.corr()[['Purchase']].sort_values('Purchase')



X = df.iloc[:,0:9]
y = df.iloc[:,9]

X = pd.get_dummies(X,columns=['Gender','Age','City_Category'])
X.head()
X= X.values

#Machine Learning

from sklearn.model_selection import train_test_split

X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.2,random_state=0)

# Appying Linear Regression to the dataset

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred =regressor.predict(X_test)


### Evaluate the model


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

rmse_test = np.sqrt(mean_squared_error(y_test,y_pred))
r2_test = r2_score(y_test,y_pred)

print("Root Mean Squared error for test set is: ", rmse_test)
print("r squared value for test set is: ", r2_test)



#Random Forest

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators =10, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


from sklearn.metrics import r2_score

r2 = r2_score(y_test,y_pred)

print("r squared value for test set is: ", r2)



