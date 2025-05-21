import pandas as pd

data=pd.read_csv(r"C:\Users\TharunMahendra\NIT\5.MachineLearning\DataPreProcessing\Data.csv")

#splitting the data to x&y
x=data.iloc[:,:-1].values #independent data


y=data.iloc[:,3].values #dependent data


#filling missing values
from sklearn.impute import SimpleImputer
#Univariate imputer for completing missing values with simple strategies.
imputer=SimpleImputer()#Definition : SimpleImputer(*, missing_values=np.nan, strategy="mean", fill_value=None, copy=True, add_indicator=False, keep_empty_features=False)
#hyper parameter tuning if we change the system startegy to our startegy if not parameter tuning default startegy
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])


#transforming the categorical to integer
from sklearn.preprocessing import LabelEncoder #Encode target labels with value between 0 and n_classes-1.
#This transformer should be used to encode target values, i.e. y, and not the input X.
encoder_x=LabelEncoder() 
encoder_x.fit_transform(x[:,0]) #converting states to 0,1,2..
x[:,0]=encoder_x.fit_transform(x[:,0])

encoder_y=LabelEncoder()
y=encoder_y.fit_transform(y)

#splitting to tarin&test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
