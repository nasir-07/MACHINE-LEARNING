"""importing the required libraries"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas
from pandas import DataFrame
import seaborn as sns
"""importing the required dataset"""
data=load_boston()
data=pandas.DataFrame(data)
"""assinging features to x and target column"""
x=data['data']
y=pandas.DataFrame(data,columns=['target'])
"""splitting the dataset for training and splitting"""
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)
print(plt.plot(xtrain,ytrain))
sns.pairplot(data)

"""scaling the dataset"""
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
"""predicting using ml model"""
dtree=DecisionTreeRegressor()
rforest=RandomForestRegressor(n_estimators=27)
kn=KNeighborsRegressor()
model=[dtree,rforest,kn]
b=[]
"""evaluating all the ml model"""
for  i in model:
    i.fit(xtrain,ytrain)
    print(i.score(xtrain,ytrain))
    b.append(cross_val_score(i,xtrain,ytrain,cv=5))

"""hypertuning our best ml model"""
a=range(1,50)
param_grid=dict(n_estimators=a)
clf=GridSearchCV(rforest,param_grid)
clf.fit(xtrain,ytrain)
print(clf.best_score_)
print(clf.best_params_)  

"""evaluating our model"""
ypred=rforest.predict(xtest)
print('training accuracy',rforest.score(xtrain,ytrain))
print('testing accuracy',rforest.score(xtest,ytest))

    

    