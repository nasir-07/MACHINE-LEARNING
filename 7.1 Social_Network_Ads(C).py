"""importing libraries"""
import pandas
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
"""reading the required dataset"""
data=pandas.read_csv('C:/Users/nasir/Desktop/Machine Learning/7.1 Social_Network_Ads.csv.csv')
"""cleaning the dataset"""
gender=pandas.get_dummies(data['Gender'])
data=pandas.concat([data,gender],axis='columns')
data=data.drop('Gender',axis='columns')
"""assigning features to x and target variable"""
x=data.drop(['Purchased','User ID','Male'],axis='columns')
print(x.isnull().sum())
y=pandas.DataFrame(data,columns=['Purchased'])
print(y.isnull().sum())
"""splitting the dataset fro training and test"""
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
""" to estimate the knn value perfect estimator
i=1
while i<10:
 knn=KNeighborsClassifier(n_neighbors=i)
 knn.fit(xtrain,ytrain)
 print('knn score for i= ',i,knn.score(xtest,ytest))
 i=i+1
 """
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
"""creating machine learning model"""
knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(xtrain,ytrain)
print('knn score ',knn.score(xtest,ytest))
"""evaluating machine learning model"""
pred=knn.predict(xtest)
cmlr=confusion_matrix(ytest,pred)
"""hypertuning our machine learning model"""
svc=SVC()
svc.fit(xtrain,ytrain)
gridd={'c':[0.1,1,100,1000],'gamma':[1,0.1,0.001,0.0001]}
a=GridSearchCV(svc,gridd,verbose=1)
a.fit(xtrain,ytrain)
print(a.best_params_)
 
 
