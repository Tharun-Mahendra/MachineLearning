import numpy as np
import pandas as pd

#lets read the data
data = pd.read_csv(r"C:\Users\TharunMahendra\NIT\6.Algorithms\1.Regression\1.SimpleLinearRegression\2.HousePricePrediction\House_data.csv")

#lets divide the data into dependent and independent variables
X = np.array(data['sqft_living']).reshape(-1, 1)
y = np.array(data['price']) 

#lets split the data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#lets create the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#lets predict the values
predictions = model.predict(X_test)

#visualize the training data
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, model.predict(X_train), color='blue')
plt.title('House Price Prediction (Training set)')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.show()

#visualize the testing data
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, model.predict(X_train), color='blue')
plt.title('House Price Prediction (Testing set)')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.show()

#lets create the pickle file
import pickle
filename = 'HousePricePredictionModel.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)