"""importing the libraries"""
import pandas
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
sc=StandardScaler()
"""importing the dataset"""
data=pandas.read_csv('loan_train.csv')
"""cleaning the dataset"""
data=data.drop(['Loan_ID'],axis=1)
data=data.dropna()
print(data.info())
print(data.isnull().sum())
set=data.select_dtypes(include=['object'])
le=LabelEncoder()
set=set.apply(le.fit_transform)
print(set.columns)
data=data.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'Property_Area', 'Loan_Status'],axis=1)
data=pandas.concat([data,set],axis=1)
"""assigning columns to  x and target variable"""
x=data.drop(['Loan_Status'],axis=1)
y=pandas.DataFrame(data,columns=['Loan_Status'])
"""splitting the dataset for training and testing"""
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=1)
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
"""creating machine learning model """
lr=LogisticRegression()
lr.fit(xtrain,ytrain)
print('accuracy lr= ',lr.score(xtrain,ytrain))
ypred=lr.predict(xtest)
"""evaluating our machine learning model"""
cm=confusion_matrix(ytest,ypred)
cf=classification_report(ytest,ypred)
print(cf)
"""testing our machine learning model"""
test=pandas.read_csv('loan_test.csv')
test=test.drop(['Loan_ID'],axis=1)
test_set=test.select_dtypes(include=['object'])
test_set=test_set.dropna()
le=LabelEncoder()
test_set=test_set.apply(le.fit_transform)
test=test.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'Property_Area'],axis=1)
test=pandas.concat([test,test_set],axis=1)
test=test.dropna()
ytesting=lr.predict(test)
print(lr.score(test,ytesting))
