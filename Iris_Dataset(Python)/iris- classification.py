#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#Import dataset
df = pd.read_csv("E:\\Shanu\\Machine-Learning-Projects\\Iris_Dataset(Python)\\iris.csv", names=["sepal-length","sepal-width","petal-length","petal-width","class"])

#Dimensions and structure of data

df.shape
df.describe()
df.head()

#Testing for the NA values

df.isna().sum()

X = df.iloc[:,0:4].values
y= df.iloc[:,4].values

# Class distribution

df.groupby("class").size()

#Visualize the struture of data

#1..Univariate plots
df.plot(kind ="box",sharex= False, sharey= True)
plt.show()

df.hist()
plt.show()

#2..Bi-variate Plots
scatter_matrix(df)
plt.show()



#Labeling the Categorical data

encoder = LabelEncoder()
y = encoder.fit_transform(y)


# Splitting dataset into training and test set

X_train,X_test , y_train, y_test = train_test_split(X,y,test_size= 0.25,random_state=0)

##################################################################################################
#Applying Algorithms to dataset

'''Logistic Regression'''
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

confusion_matrix = confusion_matrix(y_test,y_pred)
accuracy_score = accuracy_score(y_test, y_pred)#87%
classification_report = classification_report(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier,X=X_train,y=y_train,cv=10)
accuracies.mean()  #94.44%
########################################################

'''Support Vector Machines'''

classifier = SVC(random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

confusion_matrix = confusion_matrix(y_test,y_pred)
accuracy_score = accuracy_score(y_test, y_pred) 
classification_report = classification_report(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier,X=X_train,y=y_train,cv=10)
accuracies.mean() #97%

#########################################################

'''NaiveBayes'''

classifier = GaussianNB()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

confusion_matrix = confusion_matrix(y_test,y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier,X=X_train,y=y_train,cv=10)
accuracies.mean() #93.4%

########################################################

'''K-Nearest Neighbors'''

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

confusion_matrix = confusion_matrix(y_test,y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier,X=X_train,y=y_train,cv=10)
accuracies.mean() #95.3%

########################################################

'''Decision Tree'''

classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

confusion_matrix = confusion_matrix(y_test,y_pred)
accuracy_score = accuracy_score(y_test, y_pred) 
classification_report = classification_report(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier,X=X_train,y=y_train,cv=10)
accuracies.mean() #96.1%

################## PREDICT NEW VALUES AFTER FITTING MODEL################
y_p = classifier.predict([[5,2,3,4]])

