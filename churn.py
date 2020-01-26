# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:38:21 2020

@author: toshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir("C:\\Users\\toshi\\Downloads\\deepLearning\\Deep_Learning_A_Z\\Artificial_Neural_Networks")
 
#impoting the datasets
dataset= pd.read_csv("Churn_Modelling.csv")
dataset
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])

onehotencoder=OneHotEncoder(categorical_features= [1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:, 1:]



#splitting the dataset into Training set and text set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)




#Part 2 -Now Lets make the ANN
#importing the Keras libraries and package
import keras
from keras.models import Sequential
from keras.layers import Dense








#Part 4
#evaluting the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=100)
accuracies=cross_val_score(estimator=classifier,X = X_train,y = y_train,cv=10,n_jobs=-1)