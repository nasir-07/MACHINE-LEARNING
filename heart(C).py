"""importing libraries"""
import pandas
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
"""reading the required dataset"""
data=pandas.read_csv('C:/Users/nasir/Desktop/Machine Learning/heart.csv')
print(data.columns)
print(data.isnull().sum(axis='columns'))
"""assigning features to x and target column"""
x=data.drop(['target'],axis='columns')
y=pandas.DataFrame(data,columns=['target'])
"""splitting the dataset fro training and test"""
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
"""creating and evaluating our ml model"""
"""logistic regression"""
lr=LogisticRegression()
lr.fit(xtrain,ytrain)
ypredlr=lr.predict(xtest)
lrregressions=lr.score(xtest,ytest)
cmoflr=confusion_matrix(ytest,ypredlr)
print('lr accuracy is',lrregressions)
"""decision tree"""
dtree=DecisionTreeClassifier(random_state=0)
dtree.fit(xtrain,ytrain)
ypreddtree=dtree.predict(xtest)
ydtreescore=dtree.score(xtest,ytest)
print('dtree accuracy is',ydtreescore)
"""rforest"""

rforest=RandomForestClassifier(n_estimators=15,random_state=0)
rforest.fit(xtrain,ytrain)
ypredrforest=rforest.predict(xtest)
rforestt=rforest.score(xtest,ytest)
cmofrforest=confusion_matrix(ytest,ypredrforest)
print('rforest accuracy is',rforestt)
"""kneighbours"""
kneighbours=KNeighborsClassifier(n_neighbors=7)
kneighbours.fit(xtrain,ytrain)
ypredkneighbours=kneighbours.predict(xtest)
knn=kneighbours.score(xtest,ytest)
print('knn accuracy is',knn)
model_compare=pandas.DataFrame([lrregressions,rforestt,knn])
model_compare.T.plot.bar()