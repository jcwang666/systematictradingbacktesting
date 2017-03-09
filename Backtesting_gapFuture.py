# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:09:15 2017

@author: jc
"""


WORKINGPATH = "C:/python/systematictradingbacktesting/"
import os
os.chdir(WORKINGPATH)

import datetime as dt
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
import backtesting_functions as bfs


def backtesting(symol,bigpointvalue,risk,weight,cost,Distance_):
    symol= bfs.volatility(symol,36)
    #symol= bfs.calc_atr(symol,20)
    symol = tr.opening_gap(symol,90,0.85,-1)
    symol = bfs.calc_pos2(symol,bigpointvalue,risk,weight,TrendF=False,TrendR=False,PZ=True)
    symol = symol.fillna(0)
    symol = bfs.calc_pnl2(symol,cost,bigpointvalue)
    return symol    

if __name__=="__main__":
    #DAYS_IN_YEAR=256.0
    #ROOT_DAYS_IN_YEAR=DAYS_IN_YEAR**.5
    
    #========================Setup========================
    path=".//Data_backedjusted//"
    config=pd.read_csv(path+"#instrumentconfig.csv",index_col=0)
    st_config=pd.read_csv(path+"#strategy_config.csv")
    config['Cost']=config['Cost']*1
    config['Weight']=1
    #config['Distance']=1
    riskbugdet =2000
    file_key = list(config.index)
    
    currency = ['JY','AD','CD','EC','BP','MP']
    energy = ['CL','HO','NG','XRB']
    metal = ['GC','PL','PA','HG']
    usrates = ['ZB','ZF','ZN','ZT','GE']
    eurates = ['FGBL','FGBS','FGBM','FGBX']
    westequity = ['ES','FDAX','FESX','NQ','YM']
    eastequity = ['STW','SIN','SNK','SGP','SCN']
    soft = ['SB','ZW','ZC','ZS']
    fixedo = ['R','CGB','XT','YT']

    file_key = currency+westequity+eurates+usrates
    
    #========================Get Data========================
    for symbols in file_key:
        path1= path+symbols+".csv"
        tmp=pd.read_csv(path1)
        tmp.index=tmp['Date']
        tmp.index = tmp.Date.apply(lambda x: dt.datetime.strptime(x, "%Y/%m/%d"))
        locals()[symbols]=tmp.iloc[:,1:5]
        locals()[symbols].dropna()
        
    portforlio=pd.DataFrame()
    EF = pd.DataFrame()
    
    #========================Backtesting========================
    for symbols in file_key:
        locals()[symbols] = backtesting(locals()[symbols], config.ix[symbols]['Bigpointvalue'], riskbugdet, config.ix[symbols]['Weight'], config.ix[symbols]['Cost'],config.ix[symbols]['Distance'])
        portforlio[symbols]=locals()[symbols]['PnL']
        locals()[symbols]['PnL'] = locals()[symbols]['PnL'].cumsum()   


    #========================Ploting========================
    portforlio = portforlio.fillna(0)
    #portforlio.cumsum().plot(subplots=True,figsize=(9,9),layout=(2,2))
    #portforlio['2007-08-01':].cumsum().plot(subplots=True,figsize=(9,9),layout=(2,1))
    portforlio.cumsum().plot(figsize=(9,6))
    portforlio['2007-08-01':].cumsum().plot(figsize=(9,6))
    portforlio_total = pd.DataFrame(portforlio.apply(sum,axis=1))
    portforlio_total.columns=['Portforlio']
    portforlio_total.cumsum().plot(figsize=(8,5))
    
    print(portforlio_total.cumsum().tail())
    print("Daily_Std = ", portforlio_total['Portforlio'].std())
    print("sharp_ratio = ", bfs.sharpe(portforlio_total['Portforlio']))
    #portforlio_total.resample('A',how=sum)
    GE['2007':][['Close','Forecast','Position','PnL']].plot(subplots=True,figsize=(9,12))
