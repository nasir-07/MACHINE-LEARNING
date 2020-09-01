"""importing the libraries"""
import pandas
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
"""importing the required dataset"""
data=pandas.read_csv('Breast_cancer.csv')
print(data.isnull().sum())
print(data.columns)
"""cleaning the dataset"""
data=data.drop(['Unnamed: 32','id'],axis=1)
x=data.drop(['diagnosis'],axis=1)
y=pandas.DataFrame(data,columns=['diagnosis'])
"""splitting the dataset"""
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=42)
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
"""creating machine learning model"""
"""SVC"""
sc=SVC()
sc.fit(xtrain,ytrain)
scpred=sc.predict(xtest)

"""logistic regression"""
lr=LogisticRegression()
lr.fit(xtrain,ytrain)
lrpred=sc.predict(xtest)

"""evaluating our machine learning model"""
print('the accuracy of train lr',lr.score(xtrain,ytrain))
print('the accuracy of test lr',lr.score(xtest,ytest))
cmlr=confusion_matrix(ytest,lrpred)

print('the accuracy of train sc',sc.score(xtrain,ytrain))
print('the accuracy of test sc',sc.score(xtest,ytest))
cmsvc=confusion_matrix(ytest,scpred)

