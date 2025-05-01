
# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt

#let's import the dataset
data=pd.read_csv(r"C:\Users\TharunMahendra\NIT\6.Algorithms\1.Regression\Employee-Salary.csv")

#lets divide the dataset into independent and dependent variables
x=data.iloc[:,1:2].values
y=data.iloc[:,2].values

#fiiting the SVR model to the dataset
from sklearn.svm import SVR
#-->model=SVR()
# hyperParameter tuning
model=SVR(kernel='poly',degree=5,gamma='auto')
model.fit(x,y)

#predicting a new result
prediction=model.predict([[6.5]])
print(prediction)

# Visualising the SVR results
plt.scatter(x, y, color = 'red')
plt.plot(x, model.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')    
plt.ylabel('Salary')
plt.show()