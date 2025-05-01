# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt

#let's import the dataset
data=pd.read_csv(r"C:\Users\TharunMahendra\NIT\6.Algorithms\1.Regression\Employee-Salary.csv")

#lets divide the dataset into independent and dependent variables
x=data.iloc[:,1:2].values
y=data.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
#model=DecisionTreeRegressor()
model=DecisionTreeRegressor(criterion='absolute_error',splitter='random',random_state=0)
model.fit(x, y)


prediction=model.predict([[6.5]])
prediction

# Visualising the results
plt.scatter(x, y, color = 'red')
plt.plot(x, model.predict(x), color = 'blue')
plt.title('Truth or Bluff (Decision Tree)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
