"""importing libraries"""
import pandas
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
"""reading the required dataset"""
data=pandas.read_csv('census.csv')
"""Ccleaning the dataset"""
print(data.info())
set=data.select_dtypes(include=['object'])
print(set.info())
le=LabelEncoder()
set=set.apply(le.fit_transform)
print(set.info())
a=data.select_dtypes(include=['float64'])
b=data.select_dtypes(include=['int64'])
data=pandas.concat([set,a,b],axis=1)
print(data.info())
"""assigning features to x and target variable"""
x=data.drop(['income'],axis=1)
y=pandas.DataFrame(data,columns=['income'])
"""splitting the dataset fro training and test"""
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
"""creating our ml model"""
rf=RandomForestClassifier(n_estimators=100)
ad=AdaBoostClassifier()
dt=DecisionTreeClassifier()
lr=LogisticRegression()
scm=SVC()
knn=KNeighborsClassifier(n_neighbors=5)
"""finding our best machine learning model"""
a=[rf,lr,ad,dt,scm,knn]
for i in a:
 i.fit(xtrain,ytrain)
 print('the accuracy ',i, i.score(xtrain,ytrain)) 
 ypred=i.predict(xtest)
"""evaluating our machine learning model"""
 cm=confusion_matrix(ytest,ypred)
 print(cm)
 print(classification_report(ytest,ypred))
