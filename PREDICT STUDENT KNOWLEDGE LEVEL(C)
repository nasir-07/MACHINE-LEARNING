"""importing libraries"""
import pandas
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
"""reading the required dataset"""
data=pandas.read_csv('C:/Users/nasir/Desktop/Machine Learning/Data_User_Modeling_Dataset - Training_Data(TRAIN).csv')
print(data.columns)
"""cleaning the dataset"""
data=data.drop(['Attribute Information:','Unnamed: 6','Unnamed: 7'],axis='columns')
data1=data.select_dtypes(include=['object'])
le=LabelEncoder()
data1=data1.apply(le.fit_transform)
data2=data.select_dtypes(include=['float64'])
data=pandas.concat([data1,data2],axis=1)
"""assigning features to x and target variable"""
x=data.drop([' UNS'],axis=1)
y=pandas.DataFrame(data,columns=[' UNS'])
"""splitting the dataset fro training and test"""
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=1)
"""creating our ml model"""
lr=LogisticRegression()
dtree=DecisionTreeClassifier()
rforest=RandomForestClassifier()
rforest.fit(xtrain,ytrain)
knn=KNeighborsClassifier()
"""finding the best machine learning model"""
model=[lr,dtree,rforest,knn]
a=[]
for i in model:
    i.fit(xtrain,ytrain)
    a.append(cross_val_score(i,xtrain,ytrain,cv=5))
"""hypertuning our best machine learning model"""
d=range(1,30)
f=['gini','entropy']
param_grid=dict(n_estimators=d,criterion=f)
clf=GridSearchCV(rforest,param_grid)
clf.fit(xtrain,ytrain)
print(clf.best_params_)
print(clf.best_score_)
