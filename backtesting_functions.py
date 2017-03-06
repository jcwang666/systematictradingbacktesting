# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:20:26 2017

@author: rayld
"""

import datetime as dt
import numpy as np
import pandas as pd
import trading_rules as tr

DAYS_IN_YEAR=256.0
ROOT_DAYS_IN_YEAR=DAYS_IN_YEAR**.5

def weighted_sum(x, window):
    return (x*np.arange(1,window+1)).sum()

def weighted_mean(x, window):
    return (x*np.arange(1,window+1)).mean()

def df_linear_reg(df, window):
    """
    Variable:
        df = pandas Series/DataFrame
        window = rolling window for the regression

    Return:
        a tuple of two dataframes (slope, intercept)
    """
    tmp = (window*(window+1)*(window-1)/12)
    slopee = lambda x: weighted_sum(x, window ) #- (window + 1)/2.0 * pd.rolling_sum(x,window) ) / tmp
    
    #aa=df.rolling(window=window).apply(slopee)
    #aa=1
    aa=pd.rolling_apply(df,  window, slopee)
    aa = (aa -(window + 1)/2.0 * pd.rolling_sum(df,window) ) /tmp
    return aa#, intercept

def hurst(p):  
    tau = []; lagvec = []  
    #  Step through the different lags  
    for lag in range(2,20):  
        #  produce price difference with lag  
        pp = np.subtract(p[lag:],p[:-lag])  
        #  Write the different lags into a vector  
        lagvec.append(lag)  
        #  Calculate the variance of the differnce vector  
        tau.append(np.sqrt(np.std(pp)))  
    #  linear fit to double-log graph (gives power)  
    m = np.polyfit(np.log10(lagvec),np.log10(tau),1)  
    # calculate hurst  
    hurst = m[0]*2  
    # plot lag vs variance  
    #py.plot(lagvec,tau,'o'); show()  
    return hurst      
    
def volatility(price, vol_lookback):
    price['volatility'] = 0
    price['volatility'] = pd.ewmstd((price['Close'] - price['Close'].shift(1)), span=vol_lookback)
    price['volatility'] = price['volatility'].fillna(0)
    return price 

def get_carry_data(price,name,path_=".//Data_backedjusted//"):
    path1= path_+'carrydata/'+name+"1.csv"
    path2= path_+'carrydata/'+name+"2.csv"
    tmp1=pd.read_csv(path1)
    tmp2=pd.read_csv(path2)
    tmp1.index = tmp1['Date']
    tmp2.index = tmp2['Date']
    tmp1.index = tmp1.Date.apply(lambda x: dt.datetime.strptime(x, "%Y/%m/%d"))
    tmp2.index = tmp2.Date.apply(lambda x: dt.datetime.strptime(x, "%Y/%m/%d"))
    tmp1 = tmp1.iloc[:,-2]
    tmp1 = pd.DataFrame(tmp1)
    tmp1.columns=['near']
    tmp2 = tmp2.iloc[:,4]
    tmp2 = pd.DataFrame(tmp2)
    tmp2.columns=['far']
    #print(price)
    price= pd.concat([price, tmp1], axis=1, join_axes=[price.index])
    price = pd.concat([price, tmp2], axis=1, join_axes=[price.index])
    return price
    
    
def calc_atr(price, vol_lookback=20):
    tmp = pd.DataFrame()
    tmp = price
    tmp['atr1'] = abs(tmp['High']-tmp['Low'])
    tmp['atr2'] = abs(tmp['High']-tmp['Close'].shift(1))
    tmp['atr3'] = abs(tmp['Low']-tmp['Close'].shift(1))
    tmp['tr']= tmp[['atr1', 'atr2', 'atr3']].max(axis=1)
    tmp['atr'] = pd.rolling_mean(tmp['tr'],vol_lookback)
    #price['volatility'] = pd.ewma(np.maximum(abs(price['High']-price['Low']),abs(price['High']-price['Close'].shift(1)),abs(price['Low']-price['Close'].shift(1))),vol_lookback)
    price['volatility'] = tmp['atr'] 
    return price

def calc_efratio(price,window=40):    
    price['efratio']=0
    price['efratio'] = (abs(price['Close']-price['Close'].shift(window))) / pd.rolling_sum(abs(price['Close']-price['Close'].shift(1)),window)
    return price
    
    
def calc_trendfactor(price,leng=20,xx=0.3):
    tmp = pd.Series()
    price['TF'] = 0
    #vol = price['volatility']
    tmp = pd.rolling_mean(abs(pd.rolling_mean(price['Close'],leng)-pd.rolling_mean(price['Close'],leng).shift(1)),20)
    tmp = tmp / price['volatility']
    tmp = 1+(np.log(tmp+xx))
    tmp = tmp / pd.rolling_mean(tmp,200)
    tmp =tmp
    
    price['TF']=tmp
    #price['TF'][price.TF>1.5]=1.5
    #price['TF'][price.TF<0.5]=0.5
    return price
    #price['TF'] = pd.rolling_std(abs(pd.rolling_mean(price['Close'],leng)-pd.rolling_mean(price['Close'],leng).shift(1)),leng)

def calc_trendrisk(price,len1,len2):
    tmp1 = pd.Series()
    tmp2 = pd.Series()
    price['TR'] = 0
    tmp1=np.sqrt(abs((price['Close'] - pd.ewma(price['Close'],len1) )/price['volatility']))
    tmp2=np.sqrt(abs((price['Close'] - pd.ewma(price['Close'],len2) )/price['volatility']))
    tmp1 = tmp1/pd.rolling_mean(tmp1,250)
    tmp2 = tmp2/pd.rolling_mean(tmp2,250)
    price['TR'] = 1/(pd.ewma((tmp1*0.5+tmp2*0.5),10))
    price['TR'] = price['TR']*price['TR']
    #price['TR'][price.TR>1.5]=1.5
    #price['TR'][price.TR<0.5]=0.5
    return price

def calc_forecast(price,strategy,Distance,carry):  
    #config=pd.read_csv(path+"#instrumentconfig.csv",index_col=0)
    #print('tttt')
    price['Forecast'] = 0 
    if strategy == 'ewma':
        emwa_8 = tr.calc_ewmac_forecast(price,Lfast=8,usescalar=False)
        emwa_16= tr.calc_ewmac_forecast(price,Lfast=16,usescalar=False)
        emwa_32= tr.calc_ewmac_forecast(price,Lfast=32,usescalar=False)
        emwa_64= tr.calc_ewmac_forecast(price,Lfast=32,usescalar=False)
        price['Forecast'] = 0*emwa_8+3.75*emwa_16+2.65*emwa_32+1.87*emwa_64
        #print(emwa_16.head())
    elif strategy == 'break':
        bk_20 = tr.breakout(price,20,smooth=5)
        bk_40 = tr.breakout(price,40,smooth=10)
        bk_80 = tr.breakout(price,80,smooth=20)
        bk_160 = tr.breakout(price,160,smooth=40)
        bk_320 = tr.breakout(price,320,smooth=80)
        price['Forecast'] = 0.0*bk_20+0.8*bk_40+0.8*bk_80+0.8*bk_160+0.8*bk_320
    elif strategy == 'linear':
        lr_10 = tr.cal_linear_reg_forecast(price,10)
        lr_20 = tr.cal_linear_reg_forecast(price,20)
        lr_40 = tr.cal_linear_reg_forecast(price,40)
        lr_80 = tr.cal_linear_reg_forecast(price,80)
        lr_160 = tr.cal_linear_reg_forecast(price,160)
        lr_320 = tr.cal_linear_reg_forecast(price,320)
        #k_320 = tr.cal_linear_reg_forecast(price,320,smooth=80)
        price['Forecast'] = 0.0*lr_10+0.0*lr_20+0.0*lr_40+0.8*lr_80+1*lr_160+1.2*lr_320    
    elif strategy == 'carry':
        price['Forecast'] = tr.calc_carry_forecast(price,Distance)
        #price['Forecast'] = price['Forecast']*0.5
    
    if carry ==True :
        price['Forecast'] = 0.5*price['Forecast']+0.5*tr.calc_carry_forecast(price,Distance)
    
    price['Forecast'][price.Forecast>20]=20
    price['Forecast'][price.Forecast<-20]=-20
    return price
    
def calc_pos(price,bigpointvalue,risk,weight,TrendF=False,TrendR=False,PZ=True):
    if PZ :
        price['Position'] = price['Forecast']*(risk*weight)/(price['volatility']*bigpointvalue)
    else:
        price['Position'] = price['Forecast']
        
    if TrendF:
        price['Position'] = price['Position']*price['TF']
    if TrendR:
        price['Position'] = price['Position']*price['TR']
    price['Position'] = np.round(price['Position'],0)
    price['Position'] = price['Position'].shift(1)
    return price

def calc_pos2(price,bigpointvalue,risk,weight,TrendF=False,TrendR=False,PZ=True):
    if PZ :
        price['Position'] = price['Forecast']*(risk*weight)/(price['volatility'].shift(0)*bigpointvalue)
    else:
        price['Position'] = price['Forecast']
        
    if TrendF:
        price['Position'] = price['Position']*price['TF']
    if TrendR:
        price['Position'] = price['Position']*price['TR']
    price['Position'] = np.round(price['Position'],0)
    #price['Position'] = price['Position'].shift(1)
    price['Position'] = price['Position'].fillna(0)
    return price

def calc_pnl(price,cost,bigpointvalue):
    price['PnL'] = 0
    price['Cost'] = 0
    '''
    for i in range(1,len(price)):
        ##price['PnL'][i] = price['Position'][i]*(price['Close'][i]-price['Open'][i])
        price.loc[price.index[i],'PnL'] = bigpointvalue*price.loc[price.index[i],'Position']*(price.loc[price.index[i],'Close']-price.loc[price.index[i],'Open'])\
        +bigpointvalue*price.loc[price.index[i-1],'Position']*(price.loc[price.index[i],'Open']-price.loc[price.index[i-1],'Close'])\
        - abs(price.loc[price.index[i],'Position']-price.loc[price.index[i-1],'Position'])*cost
    '''
    price['PnL'] = bigpointvalue*price['Position']*(price['Close'] - price['Open'])\
    +bigpointvalue*price['Position'].shift(1)*(price['Open']-price['Close'].shift(1))\
    -abs(price['Position']-price['Position'].shift(1))*cost
    price['Cost'] = -abs(price['Position']-price['Position'].shift(1))*cost
    return price

def calc_pnl2(price,cost,bigpointvalue):
    price['PnL'] = 0
    price['Cost'] = 0
    '''
    for i in range(1,len(price)):
        ##price['PnL'][i] = price['Position'][i]*(price['Close'][i]-price['Open'][i])
        price.loc[price.index[i],'PnL'] = bigpointvalue*price.loc[price.index[i],'Position']*(price.loc[price.index[i],'Close']-price.loc[price.index[i],'Open'])\
        +bigpointvalue*price.loc[price.index[i-1],'Position']*(price.loc[price.index[i],'Open']-price.loc[price.index[i-1],'Close'])\
        - abs(price.loc[price.index[i],'Position']-price.loc[price.index[i-1],'Position'])*cost
    '''
    #price['PnL'] = bigpointvalue*price['Position']*(price['Open'] - price['Open'].shift(1))\
    #-abs(price['Position'])*cost*2
    price['PnL'] = bigpointvalue*price['Position']*(price['Close'] - price['Open'])\
    -abs(price['Position'])*cost*2
    price['Cost'] = -abs(price['Position'])*cost*2
    return price

def calc_pos_stock(price,bigpointvalue,risk,weight,TrendF=False,TrendR=False,PZ=True):
    if PZ :
        price['Position'] = price['Forecast']*(risk*weight)/(price['volatility']*bigpointvalue)
    else:
        price['Position'] = price['Forecast']
        
    if TrendF:
        price['Position'] = price['Position']*price['TF']
    if TrendR:
        price['Position'] = price['Position']*price['TR']
    price['Position'] = np.round(price['Position'],0)
    price['Position'] = price['Position'].shift(1)
    price['Position'] = price['Position'].fillna(0)
    return price    
    
    
def calc_pnl_stock(price,cost,bigpointvalue):
    price['PnL'] = 0
    price['Cost'] = 0
    '''
    for i in range(1,len(price)):
        ##price['PnL'][i] = price['Position'][i]*(price['Close'][i]-price['Open'][i])
        price.loc[price.index[i],'PnL'] = bigpointvalue*price.loc[price.index[i],'Position']*(price.loc[price.index[i],'Close']-price.loc[price.index[i],'Open'])\
        +bigpointvalue*price.loc[price.index[i-1],'Position']*(price.loc[price.index[i],'Open']-price.loc[price.index[i-1],'Close'])\
        - abs(price.loc[price.index[i],'Position']-price.loc[price.index[i-1],'Position'])*cost
    '''
    price['PnL'] = bigpointvalue*price['Position']*(price['Open'] - price['Open'].shift(1))\
    -abs(price['Position'])*cost*2*bigpointvalue*price['Open']
    price['Cost'] = -abs(price['Position'])*cost*2*bigpointvalue*price['Open']
    return price
    
def sharp_ratio(pnl):
    sharp = pnl.cumsum()[len(pnl)-1] / ( pnl.std() * np.sqrt(len(pnl))  )  
    return sharp

def annualised_rets(total_rets):
    mean_rets=total_rets.mean(skipna=True)
    annualised_rets=mean_rets*DAYS_IN_YEAR
    return annualised_rets

def annualised_vol(total_rets):
    actual_total_daily_vol=total_rets.std(skipna=True)
    actual_total_annual_vol=actual_total_daily_vol*ROOT_DAYS_IN_YEAR
    return actual_total_annual_vol

def sharpe(total_rets):
    
    sharpe=annualised_rets(total_rets)/annualised_vol(total_rets)
    
    return sharpe    
    
def drawdown(pnl):
    dd = (pnl.cummax()-pnl) * -1    
    return dd

def backtesting(symol,bigpointvalue,risk,weight,cost,Distance_,strategy,carry):
    symol= volatility(symol,36)
    #symol  = calc_atr(symol,10)
    #symol['Forecast'] = tr.calc_ewmac_forecast(symol,Lfast=16,usescalar=False)
    #symol['Forecast'] = tr.breakout(symol,256,smooth=64)
    #symol['Forecast'] = tr.calc_mac_forecast(symol,Lfast=64,usescalar=False)
    #symol['Forecast'] = tr.momentum(symol,250)*10
    #symol['Forecast'] = tr.cal_linear_reg_forecast(symol,80)
    symol = calc_forecast(symol,strategy,Distance_,carry)
    symol = calc_trendfactor(symol,20,0.3)
    symol = calc_trendrisk(symol,20,60)
    symol = calc_pos(symol,bigpointvalue,risk,weight,TrendF=False,TrendR=False,PZ=True)
    symol = symol.fillna(0)
    symol = calc_pnl(symol,cost,bigpointvalue)
    #symol['PnL'] = symol['PnL'].cumsum()
    return symol        
    