#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:18:02 2020

@author: nicolai
"""



import numpy as np
import pandas
from statsmodels.tsa.stattools import adfuller




def ADF_stationarity_test(signal, significance_level=0.05, print_results = False):
    #Dickey-Fuller test:
    adf_test = adfuller(signal, autolag='AIC')
    
    pvalue = adf_test[1]

    is_stationary = True if (pvalue<significance_level) else False
    
    if print_results:
        
        df_results = pandas.Series(adf_test[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])
        #Add Critical Values
        for key,value in adf_test[4].items():
            df_results['Critical Value (%s)'%key] = value
        print('Augmented Dickey-Fuller Test Results:')
        print(df_results)
        
    return pvalue, is_stationary



#            
