# Marks-prediction-with-ML-
marks prediction with ML models basically training a dataset with models to get the prediction whether a student is going to pass or not
#code 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,mean_squared_error
data=pd.read_csv("/content/sample_data/archive (1).zip")
df=pd.DataFrame(data)
le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df['pass']=np.where(df['How many hour do you study daily?']>3, 'passed', 'failed')
display(df)
x=df[['How many times do you seat for study in a day?','How many hour do you study daily?']]
# Encode the 'pass' column to numerical values for model training
le_pass = LabelEncoder()
y_encoded = le_pass.fit_transform(df['pass'])

# Use the encoded target variable 'y_encoded' for splitting
x_train,x_test,y_train,y_test=train_test_split(x,y_encoded,test_size=0.2,random_state=0)
lr=LinearRegression()
lr.fit(x_train,y_train)
z=int(input("how many times do you seat for study in a day?"))
w=int(input("how many hour do you study daily?"))
# Reshape the inputs z and w into a 2D array matching the format used for training
y_pred=lr.predict(np.array([[z,w]]))
print(y_pred)
if(y_pred>0.5):
  print("passed")
else:
  print("failed")
