"""importing libraries"""
import pandas
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
"""reading the dataset"""
data=pandas.read_csv('C:/Users/nasir/Desktop/Machine Learning/9.1 Salary_Data.csv.csv')
print(data.columns)
#print(data.isnull().sum())
"""assigning features to x and target variable"""
x=pandas.DataFrame(data,columns=['YearsExperience'])
y=data.drop(['YearsExperience'],axis='columns')
"""splitting the dataset for training and testing dataset"""
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
"""creating machine learning model"""
lr=LinearRegression()
lr.fit(xtrain,ytrain)
"""plotting the dataset"""
plt.plot(xtrain,ytrain)
"""evaluating our machine learning model"""
ypred=lr.predict(xtest)
print('the accuracy=',lr.score(xtest,ytest))