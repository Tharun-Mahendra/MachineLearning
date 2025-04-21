import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\TharunMahendra\NIT\6.Algorithms\1.Regression\1.SimpleLinearRegression\Salary_Data.csv")

#lets divide this data into x&y
x=df.iloc[:,0:1] #independent variable
y=df.iloc[:,1:2] #dependent variable

#lets divide data to test&train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#lets build model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_test,y_test)

#lets see the accuracy of model
predicted_data=model.predict(x_test)

#lets build some graphs
plt.scatter(x_test,y_test,color='red') #original testing data
plt.plot(x_train,model.predict(x_train),color='magenta') #regression line for training data
plt.show()

#lets predict future
#equation->y=mx+c
m=model.coef_ #slope also called coefficient
c=model.intercept_ #constant also called intercept

#if we have 25yrs exp lets predict the salary
_25yrs_exp=(m*25)+c
print(_25yrs_exp)

#lets check 10 yrs
_10yrs=(m*10)+c
print(_10yrs)
