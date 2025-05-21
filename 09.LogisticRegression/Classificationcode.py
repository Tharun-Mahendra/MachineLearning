#lets import libraries
import pandas as pd
import matplotlib.pyplot as plt

#lets read the dataset
data=pd.read_csv(r"C:\Users\TharunMahendra\NIT\6.Algorithms\2.Classification\ClassificationData.csv")

#lets divide them into dependent & independent
x=data.iloc[:,2:4] #age,salary
y=data.iloc[:,-1] #purchased

#splitting data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler #range between-> -3to3
featurescaling=StandardScaler()
x_train=featurescaling.fit_transform(x_train)
x_test=featurescaling.transform(x_test)

'''
from sklearn.preprocessing import Normalizer #range between-> 0to1
featurescaling=Normalizer()
x_train=featurescaling.fit_transform(x_train)
x_test=featurescaling.transform(x_test)
'''

#############################################################################################################################

#model building
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
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

#############################################################################################################################

#plotting graph
from sklearn.metrics import roc_curve, roc_auc_score
# Get predicted probabilities for the positive class (usually class 1)
y_prob = model.predict_proba(x_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
print("AUC Score:", auc_score)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend()
plt.grid()
plt.show()

#############################################################################################################################

#Let’s Predict Future Based on Data
PredictionData=pd.read_csv(r"C:\Users\TharunMahendra\NIT\6.Algorithms\2.Classification\FutureTestingData.csv")
FutureData=PredictionData.copy()
PredictionData=PredictionData.iloc[:,3:5]
PredictionData=featurescaling.fit_transform(PredictionData)

FuturePrediction=pd.DataFrame()
FutureData['FuturePrediction']=model.predict(PredictionData)
FutureData.to_csv('PredictedData.csv')





















