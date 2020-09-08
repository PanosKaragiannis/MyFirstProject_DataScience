# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:48:15 2019

@author: ylb18188
"""
#This demonstrates the use of Convolution1D
#Gets to 0.926 test accuracy after 2 epochs. 
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

# set parameters:
max_features = 4000 #5000
maxlen = 12 #200
batch_size = 128 #32
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
column_names_to_normalize = ['age']
x = udv[column_names_to_normalize].values
x_normscaled = preprocessing.normalize(x)
udv_temp = pd.DataFrame(x_normscaled, columns=column_names_to_normalize, index = udv.index)
udv[column_names_to_normalize] = udv_temp

#Standardization of column num_contacts
x = udv[['num_contacts']].values
scaler = StandardScaler()
x_stscaled = scaler.fit_transform(x)
udv_temp = pd.DataFrame(x_stscaled, columns=['num_contacts'], index = udv.index)
udv[['num_contacts']] = udv_temp

#Normalization with Min-Max of column created_month
x = udv[['created_month']].values
min_max_scaler = preprocessing.MinMaxScaler() 
x_minmaxscaled = min_max_scaler.fit_transform(x)
udv_temp = pd.DataFrame(x_minmaxscaled, columns=['created_month'], index = udv.index)
udv[['created_month']] = udv_temp

udv.info()


#Making the x and y sets
x = udv.drop(columns=['plan']).values
x_columns = udv.drop(columns=['plan']).columns
y = udv['plan'].values

#Min-Max Normalization
#min_max_scaler = preprocessing.MinMaxScaler()
#x = min_max_scaler.fit_transform(x)

#Split into x_train, x_test, y_train and y_test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

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

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy','binary_accuracy', f1_m,precision_m, recall_m])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

score, acc, binacc, f1_m,precision_m, recall_m = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
print('Test binary accuracy:', binacc)
print('Test f1 score:', f1_m)
print('Test precission:', precision_m)
print('Test recall:', recall_m)

score, acc, binacc, f1_m,precision_m, recall_m = model.evaluate(x_train, y_train,
                            batch_size=batch_size)
print('Train score:', score)
print('Train accuracy:', acc)
print('Test binary accuracy:', binacc)
print('Test f1 score:', f1_m)
print('Test precission:', precision_m)
print('Test recall:', recall_m)


from sklearn.metrics import classification_report
y_pred = model.predict(x_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))

##model.save('cnn_users.ipynb')