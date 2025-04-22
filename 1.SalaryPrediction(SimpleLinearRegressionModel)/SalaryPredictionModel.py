import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

#lets load the data
df=pd.read_csv(r"C:\Users\TharunMahendra\NIT\6.Algorithms\1.Regression\1.SimpleLinearRegression\SalaryPrediction\Salary_Data.csv")

#lets divide this data into x&y
x=df.iloc[:,:-1].values #independent variable
y=df.iloc[:,1].values #dependent variable

#lets divide data to test&train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#lets build model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

#lets see the accuracy of model
predicted_data=model.predict(x_test)

#lets build some graphs
#visualizing training set results
plt.scatter(x_train,y_train,color='red') #original training data    
plt.plot(x_train,model.predict(x_train),color='blue') #regression line for training data
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualizing testing set results
plt.scatter(x_test,y_test,color='red') #original testing data
plt.plot(x_train,model.predict(x_train),color='magenta') #regression line for training data
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#lets predict future
#equation->y=mx+c
m=model.coef_ #slope also called coefficient
c=model.intercept_ #constant also called intercept

#if we have 25yrs exp lets predict the salary
exp_25yrs=(m*25)+c
print(exp_25yrs)

#lets check 10 yrs
exp_10yrs=(m*10)+c
print(exp_10yrs)

#04-22


# Check model performance
bias = model.score(x_train, y_train)
variance = model.score(x_test, y_test)

from sklearn.metrics import mean_squared_error
# Calculate Mean Squared Error (MSE) for training and testing sets
train_mse = mean_squared_error(y_train, model.predict(x_train))
test_mse = mean_squared_error(y_test, predicted_data)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

#statistics for machine learning
#mean
print(df.mean()) #mean of all columns
print(df['Salary'].mean()) #mean of salary column

#median
print(df.median()) #median of all columns
print(df['Salary'].median()) #median of salary column

#mode
print(df['Salary'].mode()) #mode of salary column

#describe
print(df.describe()) #summary statistics of all columns

#variance
print(df.var()) #variance of all columns

#standard deviation
print(df.std()) #standard deviation of all columns

#correlation
print(df.corr()) #correlation matrix of all columns

#SSR
y_mean = np.mean(y) #mean of y
SSR = np.sum((predicted_data - y_mean) ** 2)
print(f"Sum of Squares Regression (SSR): {SSR:.2f}")

#SSE
y=y[0:6] #taking first 6 values of y
SSE = np.sum((y - predicted_data) ** 2)
print(f"Sum of Squares Error (SSE): {SSE:.2f}")

#SST
mean_total = np.mean(df['Salary'].values) #mean of salary values in the dataframe
SST= np.sum((df.values - mean_total) ** 2)
print(f"Total Sum of Squares (SST): {SST:.2f}")

#R-squared
r_squared = 1 - (SSE / SST)
print(f"R-squared: {r_squared:.2f}")

# Save the model to disk
filename = 'salary_prediction_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
print("Model has been pickled and saved as salary_prediction_model.pkl")
