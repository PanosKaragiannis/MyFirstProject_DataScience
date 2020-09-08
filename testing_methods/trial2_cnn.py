# -*- coding: utf-8 -*-
"""
Created on Sun May 31 02:50:36 2020

@author: ioann
"""

from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
import pandas as pd
from sklearn.model_selection import train_test_split
#from keras import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np
from keras import backend as K
from imblearn.over_sampling import SMOTE
import tensorflow as tf

# set parameters:
max_features = 4000 #5000
batch_size = 64 #32
embedding_dims = 50
filters = 50 #50
kernel_size = 1  #2
hidden_dims = 50 #50
epochs = 2

#Loading the data
udv= pd.read_csv('C:\\Users\\ioann\\Desktop\\pf-ds-coh-team2\\processed_data\\users_devices.csv')

#Drop the columns that we do not need
udv.drop('user_id', axis=1, inplace=True)
udv.drop('country', axis=1, inplace=True)
udv.drop('city', axis=1, inplace=True)
udv.drop('created_date', axis=1, inplace=True)
udv.drop('num_referrals', axis=1, inplace=True)
udv.drop('num_successful_referrals', axis=1, inplace=True)
udv.drop('created_year', axis=1, inplace=True)
udv.drop('year_2018', axis=1, inplace=True)
udv.drop('year_2019', axis=1, inplace=True)

#Normalization of column age
#column_names_to_normalize = ['age']
#x = udv[column_names_to_normalize].values
#x_normscaled = preprocessing.normalize(x)
#udv_temp = pd.DataFrame(x_normscaled, columns=column_names_to_normalize, index = udv.index)
#udv[column_names_to_normalize] = udv_temp

#Standardization of column num_contacts
#column_names_to_normalize = ['num_contacts']
#x = udv[[column_names_to_normalize]].values
#scaler = StandardScaler()
#x_stscaled = scaler.fit_transform(x)
#udv_temp = pd.DataFrame(x_stscaled, columns=[column_names_to_normalize], index = udv.index)
#udv[[column_names_to_normalize]] = udv_temp

#Normalization with Min-Max of column created_month
#min_max_scaler = preprocessing.MinMaxScaler() 
#x_minmaxscaled = min_max_scaler.fit_transform(x)
#udv_temp = pd.DataFrame(x_minmaxscaled, columns=['created_month'], index = udv.index)
#udv[['created_month']] = udv_temp

udv.info()


#Making the x and y sets
x = udv.drop(columns=['plan']).values
x_columns = udv.drop(columns=['plan']).columns
y = udv['plan'].values



#x_resample,y_resample=SMOTE().fit_sample(x,y.ravel())
#y_resample=pd.DataFrame(y_resample)
#x_resample=pd.DataFrame(x_resample)


#Split into x_train, x_test, y_train and y_test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

ssc=StandardScaler()
ssc.fit_transform(x_train)
ssc.fit_transform(x_test)

x_resample,y_resample=SMOTE().fit_sample(x_train,y_train.ravel())
y_train=pd.DataFrame(y_resample)
x_train=pd.DataFrame(x_resample)

print("X train: ", x_train.shape, "\n",
      "y train: ", y_train.shape, "\n",
      "X test: ", x_test.shape, "\n",
      "y test: ", y_test.shape, "\n")

print('Pad sequences (samples x time)')
#x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
#x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

input_dim = x_train.shape[1]
nb_classes = y_train.shape[1]


print('Build model...')
model = Sequential([
     Dense(units=max_features, input_dim=input_dim,kernel_initializer="glorot_uniform",
            activation='relu'),
     Dropout(0.2, noise_shape=None, seed=None),
     Dense(4000,activation='relu'),
     Dense(nb_classes,activation='sigmoid')  
])

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy','binary_accuracy'])

history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

score, acc, binacc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
print('Test binary accuracy:', binacc)

score, acc, binacc = model.evaluate(x_train, y_train, batch_size=batch_size)
print('Train score:', score)
print('Train accuracy:', acc)
print('Test binary accuracy:', binacc)

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

from sklearn.metrics import classification_report
y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))