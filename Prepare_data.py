import os
import time
import unicodedata
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from pandas_datareader import data as pdr

class Prepare_data(object):

    def __init__(self):
        self.file_path = './NIFTY.csv'
        self.data = pd.read_csv(self.file_path, index_col='Date')
        
        self.data = self.data.replace('null',np.nan).fillna(0).astype('float')
        
        self.preprocessed_data = None
        
        days = 20
        self.data = self.CCI(self.data, days)

        days = 14
        self.data = self.EVM(self.data, days)

        days_list = [10,50,100,200]
        for days in days_list:
            self.data = self.SMA(self.data,days)
            self.data = self.EWMA(self.data,days)

        days = 5
        self.data = self.ROC(self.data,days)

        days = 50
        self.data = self.bbands(self.data, days)

        days = 1
        self.data = self.ForceIndex(self.data,days)
       
        self.preprocessed_data = self.rescale(self.data)


    def CCI(self, data, days):
        TP = (data['High'] + data['Low'] + data['Close']) / 3
        CCI = pd.Series((TP - TP.rolling(days).mean()) / (0.015 * TP.rolling(days).std()),
        name = 'CCI')
        data = data.join(CCI)
        return data

    def EVM(self, data, days): 
        dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
        br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
        EVM = dm / br 
        EVM_MA = pd.Series(EVM.rolling(days).mean(), name = 'EVM') 
        data = data.join(EVM_MA) 
        return data

    def SMA(self,data, days): 
        sma = pd.Series(data.Close.rolling(days).mean(), name = 'SMA_' + str(days))
        data = data.join(sma) 
        return data

    def EWMA(self, data, days):
        ema = pd.Series(data.Close.ewm(span = days, min_periods = days - 1).mean(), 
        name = 'EWMA_' + str(days))
        data = data.join(ema) 
        return data

    def ROC(self,data,days):
        N = data['Close'].diff(days)
        D = data['Close'].shift(days)
        roc = pd.Series(N/D,name='ROW')
        data = data.join(roc)
        return data 

    def bbands(self, data, days):
        MA = data.Close.rolling(window=days).mean()
        SD = data.Close.rolling(window=days).std()
        data['UpperBB'] = MA + (2 * SD) 
        data['LowerBB'] = MA - (2 * SD)
        return data

    def ForceIndex(self, data, days): 
        FI = pd.Series(data['Close'].diff(days) * data['Volume'], name = 'ForceIndex') 
        data = data.join(FI) 
        return data

    def rescale(self,data):
        names = data.columns
        data = data.dropna().astype('float')
        scaler = sklearn.preprocessing.StandardScaler()
        scaled_df = scaler.fit_transform(data)
        data = pd.DataFrame(scaled_df,columns=names)
        print(data)
        return data
		
if __name__ == '__main__':
	prepare_data = Prepare_data()
	