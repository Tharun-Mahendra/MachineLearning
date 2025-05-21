# import Libraries
import numpy as np
import pandas as pd

# read the data
dataset=pd.read_csv(r"C:\Users\TharunMahendra\NIT\6.Algorithms\1.Regression\2.MultipleLinearRegression\1.Investment Prediction\Investment.csv")

#divide into dependent & Independent variables
x=dataset.iloc[:,0:4]
y=dataset.iloc[:,4]

# lets fill categorical variables with dummies
x=pd.get_dummies(x,dtype=int)

#divide data into test & train data in 80:20 ratio
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#lets build model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

#lets predict y_test
prediction=model.predict(x_test)

# slope
m_coefficient=model.coef_
print(m_coefficient)
# constant
c_intercept=model.intercept_
print(c_intercept)

#lets add constants to x with ones or with constant variables
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1) #--> ones


# lets import API statsmodels ,method leastsquares and call the model OLS 
import statsmodels.api as sm
x_opt=x[:,[0,1,2,3,4,5]]
#OrdinaryLeastSquares
model_OLS=sm.OLS(endog=y,exog=x_opt).fit() #endog=dependent variable ,exog=independent
model_OLS.summary()

#recuresive feature elimination(backward elimination)
#lets delete the variables having more than 0.05 p value

x_opt=x[:,[0,1,2,3,5]]
#OrdinaryLeastSquares
model_OLS=sm.OLS(endog=y,exog=x_opt).fit()
model_OLS.summary()

x_opt=x[:,[0,1,2,3]]
#OrdinaryLeastSquares
model_OLS=sm.OLS(endog=y,exog=x_opt).fit()
model_OLS.summary()

x_opt=x[:,[0,1,3]]
#OrdinaryLeastSquares
model_OLS=sm.OLS(endog=y,exog=x_opt).fit()
model_OLS.summary()

x_opt=x[:,[0,1]]
#OrdinaryLeastSquares
model_OLS=sm.OLS(endog=y,exog=x_opt).fit()
print(model_OLS.summary())

# therefore x1 i.e DigitalMarketing
# investing in DigitalMarketing can give the profits is the prediction of the model

bias=model.score(x_train,y_train)
print(bias)

regressor=model.score(x_test,y_test)
print(regressor)

















