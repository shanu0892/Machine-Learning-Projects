import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("C:\\Users\\Arun\\Documents\\shanu\\kaggle\\heart.csv")

df.isna().sum()

df.shape

plt.figure(figsize=(14,8))
sns.heatmap(df.corr(),annot = True)
plt.show()

sns.distplot(df["trestbps"])
sns.distplot(x= df["thalach"], y= df["target"])


sizes = [len(df[df['exang'] == 0]),len(df[df['exang'] == 1])]
plt.pie(sizes, labels= ['No','Yes'] )

sizes = [len(df[df['sex'] == 0]),len(df[df['sex'] == 1])]
plt.pie(sizes, labels= ['Female','Male'] ,autopct = '%.1f%%' ,center=(0,0))

sizes = [len(df[df['cp'] == 0]),len(df[df['cp'] == 1]),len(df[df['cp'] == 2]),len(df[df["cp"] == 3])]
plt.pie(sizes, labels = [0,1,2,3] ,autopct = '%.1f%%' ,center=(0,0))

sizes = [len(df[df['fbs'] ==1]),len(df[df['fbs'] ==0 ])]
plt.pie(sizes, labels= ['true','false'] ,autopct = '%.1f%%' ,center=(0,0))

sizes = [len(df[df['restecg'] == 0]),len(df[df['restecg'] == 1])]
plt.pie(sizes, labels= ['False','True'] ,autopct = '%.1f%%' ,center=(0,0))

# 68.3% Males of age greater than 50 , chest pain type = 0 ,14.9% have blood pressure greater

g= sns.FacetGrid(df,col= "sex")
g.map(sns.barplot,"fbs", "target")
# blood sugar is greater in males have target of nearly 1

plt.figure(figsize=(14,8))
sns.countplot(x= "age", data= df, hue="target")




X = df.iloc[:,0:13].values
y= df.iloc[:,13].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

################################################################################
#CLASSIFICATION



#KNN classifier

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_before = accuracy_score(y_test, y_pred) ##83.6%

from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator=classifier ,X = X_train, y = y_train, cv=10,)
accuracy.mean() ## 81.4%

######################################################
### Support Vector Machine

from sklearn.svm import SVC
classifier = SVC(random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_before = accuracy_score(y_test, y_pred)#86.9%


from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator=classifier ,X = X_train, y = y_train, cv=10,)
accuracy.mean()#### 81%




#####Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.metrics import accuracy_score
accuracy_before = accuracy_score(y_test, y_pred)#77%


from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator=classifier ,X = X_train, y = y_train, cv=10,)
accuracy.mean() # 79%

#########################
#### RandomForest classification

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



from sklearn.metrics import accuracy_score
accuracy_before = accuracy_score(y_test, y_pred)#83.6%



from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator=classifier ,X = X_train, y = y_train, cv=10,)
accuracy.mean() # 82.2%

###########################################################################################

###ROC CURVE

from sklearn.metrics import roc_auc_score, roc_curve
y_prob= classifier.predict_proba(X_test)[:,1]

fpr , tpr, thres = roc_curve(y_test,y_prob)

plt.figure(figsize= (10,6))
plt.plot(fpr,tpr)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.show()

acc = roc_auc_score(y_test, y_prob)












