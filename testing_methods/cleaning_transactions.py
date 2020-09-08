# -*- coding: utf-8 -*-
"""
Created on Sun May 31 13:11:16 2020

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

t1=pd.read_csv('C:\\Users\\ioann\\Desktop\\pf-ds-coh-team2\\data\\transactions_1.csv')
t2=pd.read_csv('C:\\Users\\ioann\\Desktop\\pf-ds-coh-team2\\data\\transactions_2.csv')
t3=pd.read_csv('C:\\Users\\ioann\\Desktop\\pf-ds-coh-team2\\data\\transactions_3.csv')
# concatenate the 3 datasets into 1 with all transactions
tdf=pd.concat([t1, t2, t3]) 

print(tdf.shape)
print(tdf.isna().sum())
print(len(np.unique(tdf['user_id']))) #how many users have made a transaction?
#print(tdf.sample(5))
tdf.info()

tdf['created_date'] = tdf['created_date'].apply(pd.to_datetime)

#Create the dummies about the years 2018 and 2019
tdf=pd.concat([pd.get_dummies(tdf['transactions_type']), tdf], axis=1)
#tdf=pd.concat([pd.get_dummies(tdf['transactions_currency']), tdf], axis=1)
tdf=pd.concat([pd.get_dummies(tdf['transactions_state']), tdf], axis=1)
tdf.info()
print(tdf.sample(5))

tdf_grouped = tdf.groupby(['user_id'], as_index=False)
#print(tdf_grouped.sample(4))

tdf_grouped_user=tdf.groupby(["user_id"])["CANCELLED", 'COMPLETED','DECLINED','FAILED','PENDING', 
                            'REVERTED', 'ATM','CARD_PAYMENT','CARD_REFUND','CASHBACK','EXCHANGE',
                            'FEE','REFUND','TAX', 'TOPUP','TRANSFER'].sum()
print(tdf_grouped_user.shape)
tdf_grouped_user.sample(5)

tdf_grouped_user2=tdf.groupby(["user_id"])["user_id"].count()
print(tdf_grouped_user2.shape)
#tdf_grouped_user2.sample(5)


tdf_grouped = pd.merge(tdf_grouped_user2, tdf_grouped_user, on='user_id', how='left')
tdf_grouped.sample(5)

#us_tr_left = pd.merge(us, tr_grouped, on='user_id', how='left')
#print(us_tr_left.shape)
#print(us_tr_left.isna().sum())
#us_tr_left.sample(5)