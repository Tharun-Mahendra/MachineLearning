

#lets import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#lets read the dataset
data=pd.read_csv(r"C:\Users\TharunMahendra\NIT\6.Algorithms\2.Classification\ClassificationData.csv")

#lets divide them into dependent & independent
x=data.iloc[:,2:4] #age,salary
y=data.iloc[:,-1] #purchased

#splitting data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

'''
#feature scaling
from sklearn.preprocessing import StandardScaler #range between-> -3to3 
featurescaling=StandardScaler()
x_train=featurescaling.fit_transform(x_train)
x_test=featurescaling.transform(x_test)
'''

from sklearn.preprocessing import Normalizer
featurescaling=Normalizer()
x_train=featurescaling.fit_transform(x_train)
x_test=featurescaling.transform(x_test)

#############################################################################################################################

#model building
from sklearn.naive_bayes  import BernoulliNB,MultinomialNB,GaussianNB
#model=BernoulliNB()
#model=GaussianNB()

#we have to use Normalizer for Multionmial Navie-Bayes as Negative Values can't be passed to Multinomial where 
# Standard Scaler Ranges from -3
model=MultinomialNB()
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