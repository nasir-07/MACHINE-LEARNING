"""importing libraries"""
import pandas
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
"""reading the required dataset"""
data=pandas.read_csv('iris.csv')
print(data.info())
"""cleaning the dataset"""
set=data.select_dtypes(include=['object'])
le=LabelEncoder()
set=set.apply(le.fit_transform)
data=data.drop(['Species'],axis=1)
data=pandas.concat([data,set],axis=1)
"""assigning features to x and target column"""
x=data.drop(['Species'],axis=1)
y=pandas.DataFrame(data,columns=['Species'])
pca=PCA(n_components=2)
pca.fit(x)
x=pca.transform(x)
"""splitting the dataset fro training and test"""
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=1)
"""creating our ml model"""
rf=RandomForestClassifier()
dtree=DecisionTreeClassifier()

"""dimension reduction(OPTIONAL-SKIP ALL THE NEXT FUNCTION IF FOLLOWING THIS FUNCTION)"""
pca=PCA(n_components=2)
pca.fit(xtrain)
xn=pca.transform(xtrain)
plt.plot(xn,ytrain)
rf.fit(xtrain,ytrain)


"""DecisionTreeClassifier"""

dtree=DecisionTreeClassifier()
dtree.fit(xtrain,ytrain)
print('accuracy dtree= ',dtree.score(xtrain,ytrain))
dtreep=dtree.predict(xtest)
cmtree=confusion_matrix(ytest,dtreep)



"""RandomForestClassifier"""
rforest=RandomForestClassifier(n_estimators=100)
rforest.fit(xtrain,ytrain)
print('accuracy rforest= ',rforest.score(xtrain,ytrain))
rforestp=rforest.predict(xtest)
cmrf=confusion_matrix(ytest,rforestp)

"""SVC"""
svc=SVC(kernel='rbf')
svc.fit(xtrain,ytrain)
print('accuracy svc= ',svc.score(xtrain,ytrain))
svcp=svc.predict(xtest)
cmsvc=confusion_matrix(ytest,svcp)

"""knn"""
knn=KNeighborsClassifier(n_neighbors=12)
knn.fit(xtrain,ytrain)
print('accuracy knn= ',knn.score(xtrain,ytrain))
knnp=knn.predict(xtest)
cmknn=confusion_matrix(ytest,knnp) 
"""finding the perfect ml model"""
model=[dtree,rforest,svc,knn]
a=[]
for i in model:
 cv=a.append(cross_val_score(i,xtrain,ytrain,cv=10))

"""hypertuning our model"""
i=range(1,32)
methoda=['rbf','linear']
param_grid=dict(C=i,kernel=methoda)
clf=GridSearchCV(SVC(),param_grid,cv=10)
clf.fit(xtrain,ytrain)
print(clf.best_score_)
print(clf.best_params_)

"""evaluating our dataset"""
ypred=rf.predict(xtest)
cm=confusion_matrix(ytest,ypred)
print(cm)
 


