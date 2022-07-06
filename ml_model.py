#importing libraries
import pandas as pd
import numpy as np
import pickle

df=pd.read_csv("campus_placement_dataset.csv")

#Feature Engineering
df=df.drop('RegNo',axis=1)

#loading the data
#method 1 - using iloc x->independent variable,y->dependent variable
X = df.iloc[:,:-1].values#independent -- message
Y = df.iloc[:,-1:].values #dependent -- label

#splitting the data nto traning and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=20)


#standing scaling - normalizing the data - x train
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

#training the model

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(x_train,y_train)


#testing
#predicting
y_pred=classifier.predict(sc.transform(x_test))
print(y_pred)


pickle.dump(classifier,open('model.pkl','wb')) #we are serializing our model by creating model.pkl and writing into it by 'wb'
model=pickle.load(open('model.pkl','rb'))
print("Sucess loaded")


#execute this file only once and create the pkl file.