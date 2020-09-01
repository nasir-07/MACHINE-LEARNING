"""importing the libraries"""
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
"""reading the dataset"""
data=pandas.read_csv('C:/Users/nasir/Desktop/Machine Learning/training_variants')
data=data.drop(['ID'],axis=1)
print(data.isnull().sum())
"""cleaning the dataset"""
set=data.select_dtypes(include=['object'])
le=LabelEncoder()
set=set.apply(le.fit_transform)
set1=data.select_dtypes(include=['int64'])
data=pandas.concat([set,set1],axis=1)
"""creating x and target columns"""
x=data.drop(['Class'],axis=1)
y=pandas.DataFrame(data,columns=['Class'])
"""splitting the dataset"""
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=1)
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
"""creating machine learning model"""
dtree=DecisionTreeClassifier()
rforest=RandomForestClassifier(n_estimators=48)
lr=LogisticRegression()
sv=SVC(C=59,kernel='rbf')
model=[dtree,rforest,lr,sv]
a=[]
"""finding the best model"""
for i in model:
    i.fit(xtrain,ytrain)
    print('the accuracy= ', i.score(xtrain,ytrain))
    a.append(cross_val_score(i,xtrain,ytrain,cv=10))
ypred=sv.predict(xtest)
"""evaluating our machine learning model"""
cm=confusion_matrix(ytest,ypred)
print(classification_report(ytest,ypred))
print(sv.score(xtest,ytest))
"""hypertuning our machine learning model"""
r=range(0,60)
method=['linear','rbf']
param_grid=dict(C=r,kernel=method)
clf=GridSearchCV(sv,param_grid)
clf.fit(xtrain,ytrain)
print(clf.best_score_)
print(clf.best_params_)
"""