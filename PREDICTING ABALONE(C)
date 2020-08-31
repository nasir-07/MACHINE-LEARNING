"""importing the required libraries"""
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
"""reading dataset"""
data=pandas.read_csv('C:/Users/nasir/Desktop/Machine Learning/abalone.csv')
#print(data.info())
"""cleaning dataset"""
data3=data.select_dtypes(include=['object'])
data1=data.select_dtypes(include=['int64'])
data2=data.select_dtypes(include=['float64'])
le=LabelEncoder()
data3=data3.apply(le.fit_transform)
data=pandas.concat([data1,data2,data3],axis=1)
"""assinging features to x and target """
x=data.drop(['Rings'],axis=1)
y=pandas.DataFrame(data,columns=['Rings'])
"""splitting the dataset for training and testing """
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=1)
"""creating machine learning model"""
lr=LogisticRegression()
dtree=DecisionTreeClassifier()
rforest=RandomForestClassifier()
knn=KNeighborsClassifier()

"""finding the best machine learning model"""
score=[]
model=[knn,dtree,rforest,svm]
for i in model:
    i.fit(xtrain,ytrain)
    score.append(cross_val_score(i,xtrain,ytrain,cv=5))

"""hypertuning our best model"""    
b=range(1,35)
param_grid=dict(n_estimators=b)
clf=GridSearchCV(rforest,param_grid)
clf.fit(xtrain,ytrain)
print(clf.best_params_)
print(clf.best_score_)
"""deploying or best ml model"""
rforest.fit(xtrain,ytrain)
"""evaluating our machine learning model"""
ypred=rforest.predict(xtest)
cm=confusion_matrix(ytest,ypred)
print(cm)
    

    
