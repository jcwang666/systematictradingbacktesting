# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:49:46 2017

@author: JC
"""


WORKINGPATH = "C:/python/systematictradingbacktesting/"
import os
os.chdir(WORKINGPATH)

import datetime as dt
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
import backtesting_functions as bfs
#import trading_rules as tr

    
if __name__=="__main__":
    DAYS_IN_YEAR=256.0
    ROOT_DAYS_IN_YEAR=DAYS_IN_YEAR**.5
    
    #========================Setup========================
    path=".//Data_backedjusted//"
    config=pd.read_csv(path+"#instrumentconfig.csv",index_col=0)
    st_config=pd.read_csv(path+"#strategy_config.csv")
    #config['Cost']=0
    #config['Weight']=1
    config['Distance']=1
    riskbugdet = 30000    
    #STRATEGY = 'break' 'ewma' 'linear'
    STRATEGY = 'ewma'   
    #CARRY = True  : 50%carry + 50%Trend Following
    #CARRY = Flase : 100%Trend Following
    CARRY = True

    file_key = list(config.index)
  
    currency = ['JY','AD','CD','EC','BP','MP']
    energy = ['CL','HO','NG','XRB']
    metal = ['GC','PL','PA','HG']
    fixed = ['ZB','ZF','ZN','ZT','GE']
    Efixed = ['FGBL','FGBS','FGBM','FGBX']
    fixedo = ['R','CGB','XT','YT']
    westequity = ['ES','FDAX','FESX','NQ','YM']
    soft = ['SB','ZW','ZC','ZS']
    eastequity = ['STW','SIN','SNK','SGP','SCN']
    file_key = ['GC','PL','CL','ZF','ZN','FGBL','FGBM','YM','ES','HO','XRB','NQ']
    file_key = currency+energy+metal+fixed+Efixed+westequity+soft
    #file_key=soft+currency+energy+metal+fixed+Efixed+westequity
    carry_key =file_key
    
    #========================Get Data========================
    for symbols in file_key:
        path1= path+symbols+".csv"
        tmp=pd.read_csv(path1)
        tmp.index=tmp['Date']
        tmp.index = tmp.Date.apply(lambda x: dt.datetime.strptime(x, "%Y/%m/%d"))
        locals()[symbols]=tmp.iloc[:,1:5]
        locals()[symbols].dropna()
    
    if CARRY == True:
        for symbols in carry_key:
            locals()[symbols] = bfs.get_carry_data(locals()[symbols],symbols)

    #========================Backtesting========================
    portforlio=pd.DataFrame()   
    EF = pd.DataFrame()
    
    for symbols in file_key:
        locals()[symbols]['weight'] = 1/len(file_key)
        locals()[symbols] = bfs.calc_efratio(locals()[symbols],150)
        #locals()[symbols] = backtesting(locals()[symbols], config.ix[symbols]['Bigpointvalue'], riskbugdet, config.ix[symbols]['Weight'], config.ix[symbols]['Cost'],config.ix[symbols]['Distance'])
        locals()[symbols] = bfs.backtesting(locals()[symbols], config.ix[symbols]['Bigpointvalue'], riskbugdet, locals()[symbols]['weight'], config.ix[symbols]['Cost'],config.ix[symbols]['Distance'],strategy=STRATEGY,carry = CARRY)
        portforlio[symbols]=locals()[symbols]['PnL']
        locals()[symbols]['Cumpnl'] = locals()[symbols]['PnL'].cumsum()   
        EF[symbols]=locals()[symbols]['efratio']
        
    #for symbols in file_key:
    #    locals()[symbols]['cumpnl'] = locals()[symbols]['PnL'].cumsum()   
        #portforlio=pd.merge(portforlio,locals()[symbols]['PnL'])
                
    #========================Ploting========================
    portforlio = portforlio.fillna(0)
    #portforlio.cumsum().plot(subplots=True,figsize=(12,15),layout=(10,3))
    portforlio.cumsum().plot(figsize=(8,5))
    portforlio_total = pd.DataFrame(portforlio.apply(sum,axis=1))
    portforlio_total.columns=['Portforlio']
    portforlio_total.cumsum().plot(figsize=(8,5))
    #print(portforlio_total.cumsum().tail())
    print("Daily_Std = ", portforlio_total['Portforlio'].std())
    print("sharp_ratio = ", bfs.sharpe(portforlio_total['Portforlio']))
    #portforlio_total.resample('A',how=sum)
    
    #========================Detail========================
    FGBL['2001':][['Close','Forecast','Position','Cumpnl']].plot(subplots=True,figsize=(9,12))
    #print(GC[['efratio','PnL']].corr())
