# We will learn slightly advanced ML model called as - POLYNOMIAL REGRESSION MODEL
#We did before was simple linear regression and multiple linear regression modeL
#so far we build a linear regressor as linear & multilinear regressor

#from now we are going to build regressor but that are not linear any more 
#polynomial regression is not linear regressor, then we build svr, then we build the decission tree regressor & random forset 
#regression model which are not linear at all
#if i use polynomial term in simple linear regression then that is called as polynomial regressor
#next svr,dt,random forest based will be based on more complex theory
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#lets import Libraries
import pandas as pd
import matplotlib.pyplot as plt

#lets read dataset
data=pd.read_csv(r"C:\Users\TharunMahendra\NIT\6.Algorithms\1.Regression\Employee-Salary.csv")

# PROBLEM STATEMENT:
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#We are the HR team of a large company preparing to hire a new employee. After interviews, we find the candidate a good fit. During salary negotiations, 
#the candidate claims they earned **$161K annually** as a **Regional Manager** in their previous role and is asking for a higher salary.
#To verify this, one HR member contacts the former employer but only receives a dataset containing **Position, Level, and Salary** for 10 roles. An analysis shows a 
# **non-linear relationship** between Level and Salary.
#We also learn that the candidate has been a **Regional Manager (Level 6)** for 2 years, and it typically takes 4 years to become a Partner (Level 7). 
# So, we estimate their level as **6.5**.
#To verify the claim, we plan to build a **polynomial regression model** using the **Level (X)** and **Salary (y)** columns from the dataset. 
# The goal is to **predict the salary for Level 6.5** and compare it to the claimed **$161K** to determine if the candidate is **truthful or bluffing**.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# DIVIDE THE DATASET INTO INDEPENDENT AND DEPENDENT VARIABLES
x=data.iloc[:,1:2].values
y=data.iloc[:,2].values
#as the data is small we can use the whole data for training and testing

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TRAINING THE MODEL USING LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
reg_model1=LinearRegression()
reg_model1.fit(x,y) #training the model

plt.scatter(x,y)
plt.plot(x,reg_model1.predict(x),color='m')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

reg_model1prediction=reg_model1.predict([[6.5]])
print(reg_model1prediction)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TRAINING THE MODEL USING POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
#parameter tuning default degree=2
# poly_model=PolynomialFeatures()
#hyper-parameter tuning lets increase degree which increases the accuracy of model
# poly_model=PolynomialFeatures(degree=3)
# poly_model=PolynomialFeatures(degree=4)
poly_model=PolynomialFeatures(degree=5)
# poly_model=PolynomialFeatures(degree=6)
x_transform=poly_model.fit_transform(x) #transforming the data into polynomial features
#x_poly will contain the original x and x^2, x^3, x^4...x^n (degree of polynomial)

reg_model2=LinearRegression()
reg_model2.fit(x_transform,y) #training the regression model

plt.scatter(x,y)
plt.plot(x,reg_model2.predict(x_transform),color='m')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

reg_model2prediction=reg_model2.predict(poly_model.fit_transform([[6.5]]))
print(reg_model2prediction)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# That means employee is True and we solved this by using polyregression model
