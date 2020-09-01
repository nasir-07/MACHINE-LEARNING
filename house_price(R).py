"""importing libraries"""
import pandas
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
"""reading the required dataset"""
data=pandas.read_csv('house_price.csv')
"""cleaning the dataset"""
data=data.drop(['Alley','PoolQC','MiscFeature','Fence','Id','FireplaceQu'],axis=1)
data=data.dropna()
print(data.info())
set=data.select_dtypes(include=['object'])
le=LabelEncoder()
set=set.apply(le.fit_transform)
print(set.columns)
data=data.drop(['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition'],axis='columns')
data=pandas.concat([data,set],axis='columns')
"""assigning features to x and target column"""
x=data.drop(['SalePrice'],axis='columns')
y=pandas.DataFrame(data,columns=['SalePrice'])
"""splitting the dataset for training and test"""
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25)
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
"""creating our ml model"""
lr=LinearRegression()
lr.fit(xtrain,ytrain)
lrpred=lr.predict(xtest)
dtree=DecisionTreeRegressor()
dtree.fit(xtrain,ytrain)
dtpred=lr.predict(xtest)

"""evaluating our machine learning model"""
print('the accuracy of train lr',lr.score(xtrain,ytrain))
print('the accuracy of test lr',lr.score(xtest,ytest))
print('the accuracy of train dtree',dtree.score(xtrain,ytrain))
print('the accuracy of test dtree',dtree.score(xtest,ytest))