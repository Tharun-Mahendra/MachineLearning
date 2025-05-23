
#lets import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#lets read the dataset
data=pd.read_csv(r"C:\Users\TharunMahendra\NIT\6.Algorithms\2.Classification\Churn_Modelling.csv")

#lets divide them into dependent & independent
x=data.iloc[:,3:13].values
y=data.iloc[:,-1].values

#Converting/Encoding categorical variables to numerical

# LabelEncoding Gender column
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])
# OneHot Encoding the Geography
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
# other columns unchanged (remainder='passthrough').
x= np.array(ct.fit_transform(x))


#splitting data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#############################################################################################################################

#model building
from xgboost import XGBClassifier
model=XGBClassifier()
model.fit(x_train,y_train)

#prediction
y_pred=model.predict(x_test)

#confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)

from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)

bias=model.score(x_train,y_train)
variance=model.score(x_test,y_test)