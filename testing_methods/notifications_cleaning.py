# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:30:02 2020

@author: ioann
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie,axis,show
import seaborn as sns
import random as rd
import datetime as dt
from termcolor import colored

ndf=pd.read_csv('C:\\Users\\ioann\\Desktop\\pf-ds-coh-team2\\data\\notifications.csv')
print('The shape of notifications is: ',ndf.shape)

print(ndf.describe())
print(ndf.dtypes)
print(ndf.isnull().sum()) #we do not have nulls in any column
print(np.unique(ndf['reason']))
print(np.unique(ndf['status']))
print(np.unique(ndf['channel']))
print(pd.crosstab(index=ndf['reason'], columns=ndf['status']))

ndf['created_date'] = ndf['created_date'].apply(pd.to_datetime)
#print(np.unique(ndf['created_date']))

ndf=pd.concat([pd.get_dummies(ndf['reason']), ndf], axis=1)
ndf=pd.concat([pd.get_dummies(ndf['status']), ndf], axis=1)
ndf=pd.concat([pd.get_dummies(ndf['channel']), ndf], axis=1)

ndf.info()


