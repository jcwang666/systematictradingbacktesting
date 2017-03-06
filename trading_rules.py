# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 11:15:17 2016

@author: rayld
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import utility_functions as utl

def calc_ewmac_forecast(price, Lfast, Lslow=None, usescalar=True):   
    """
    Calculate the ewmac trading fule forecast, given a price and EWMA speeds Lfast, Lslow and vol_lookback
    
    Assumes that 'price' is daily data
    """
    ## price: This is the stitched price series
    ## We can't use the price of the contract we're trading, or the volatility will be jumpy
    ## And we'll miss out on the rolldown. See http://qoppac.blogspot.co.uk/2015/05/systems-building-futures-rolling.html
    if Lslow is None:
        Lslow=4*Lfast
    
    ## We don't need to calculate the decay parameter, just use the span directly    
    fast_ewma=pd.ewma(price['Close'], span=Lfast)
    slow_ewma=pd.ewma(price['Close'], span=Lslow)
    raw_ewmac=fast_ewma - slow_ewma
    
    ## volatility adjustment
    stdev_returns=price['volatility']  
    vol_adj_ewmac=raw_ewmac/stdev_returns
    
    ## scaling adjustment
    if usescalar:
        f_scalar=ewmac_forecast_scalar(Lfast, Lslow)
        forecast=vol_adj_ewmac*f_scalar
    else:
        forecast=vol_adj_ewmac
    
    #cap_forecast=cap_series(forecast, capmin=-20.0,capmax=20.0)    
    return vol_adj_ewmac
    
def calc_mac_forecast(price, Lfast, Lslow=None, usescalar=True):   
    """
    Calculate the ewmac trading fule forecast, given a price and EWMA speeds Lfast, Lslow and vol_lookback
    
    Assumes that 'price' is daily data
    """
    ## price: This is the stitched price series
    ## We can't use the price of the contract we're trading, or the volatility will be jumpy
    ## And we'll miss out on the rolldown. See http://qoppac.blogspot.co.uk/2015/05/systems-building-futures-rolling.html
    if Lslow is None:
        Lslow=4*Lfast
    
    ## We don't need to calculate the decay parameter, just use the span directly    
    fast_ewma=pd.rolling_mean(price['Close'], Lfast)
    slow_ewma=pd.rolling_mean(price['Close'], Lslow)
    raw_ewmac=fast_ewma - slow_ewma
    
    ## volatility adjustment
    stdev_returns=price['volatility']  
    vol_adj_ewmac=raw_ewmac/stdev_returns
    
    ## scaling adjustment
    if usescalar:
        f_scalar=ewmac_forecast_scalar(Lfast, Lslow)
        forecast=vol_adj_ewmac*f_scalar
    else:
        forecast=vol_adj_ewmac
    
    #cap_forecast=cap_series(forecast, capmin=-20.0,capmax=20.0)    
    return vol_adj_ewmac
    
 
def momentum(price,window=252):    
    mom = np.sign(price['Close'] - price['Close'].shift(window))
    return mom    

def ma(price,window=10):    
    ma = np.sign(pd.rolling_mean(price['Close'],window) -pd.rolling_mean(price['Close'],window).shift(1))
    return ma     
    
def breakout(price, lookback, smooth=None):
    """
    :param price: The price or other series to use (assumed Tx1)
    :type price: pd.DataFrame

    :param lookback: Lookback in days
    :type lookback: int

    :param lookback: Smooth to apply in days. Must be less than lookback! Defaults to smooth/4
    :type lookback: int

    :returns: pd.DataFrame -- unscaled, uncapped forecast

    With thanks to nemo4242 on elitetrader.com for vectorisation

    """
    if smooth is None:
        smooth = max(int(lookback / 4.0), 1)

    #assert smooth < lookback

    roll_max = pd.rolling_max(price['Close'], lookback, min_periods=int(
        min(len(price['Close']), np.ceil(lookback / 2.0))))
    roll_min = pd.rolling_min(price['Close'], lookback, min_periods=int(
        min(len(price['Close']), np.ceil(lookback / 2.0))))

    #roll_max = pd.rolling_max(price['Close'], lookback)
    #roll_min = pd.rolling_min(price, lookback)
    roll_mean = (roll_max + roll_min) / 2.0

    # gives a nice natural scaling
    output = 40.0 * ((price['Close'] - roll_mean) / (roll_max - roll_min))
    smoothed_output = pd.ewma(
        output,
        span=smooth,
        min_periods=np.ceil(
            smooth / 2.0))

    return smoothed_output    
    

def calc_carry_forecast(price,Distance, f_scalar=30.0):

    """
    Carry calculation
    
    Formulation here will work whether we are trading the nearest contract or not
    
    For other asset classes you will have to work out nerpu (net expected return in price units) yourself
    """
    #print(Distance)
    #nerpu=carrydata.apply(find_datediff, axis=1)
    
    stdev_returns=price['volatility']
    ann_stdev=stdev_returns*16
    nerpu = (price['near'] - price['far'] )/(Distance/12)
    raw_carry=nerpu/ann_stdev
    
    forecast=raw_carry*f_scalar
    forecast=pd.ewma(forecast,span=2)
    #cap_forecast=cap_series(forecast)
    
    return forecast
    
    
 #=============================================   
def weighted_sum(x, window):
    return (x*np.arange(1,window+1)).sum()

def weighted_mean(x, window):
    return (x*np.arange(1,window+1)).mean()

def cal_linear_reg_forecast(price, window):

    """
    Variable:
        df = pandas Series/DataFrame
        window = rolling window for the regression

    Return:
        a tuple of two dataframes (slope, intercept)
    """
    stdev_returns=price['volatility']  
    tmp = (window*(window+1)*(window-1)/12)
    slopee = lambda x: weighted_sum(x, window ) #- (window + 1)/2.0 * pd.rolling_sum(x,window) ) / tmp
    
    #aa=df.rolling(window=window).apply(slopee)
    #aa=1
    aa=pd.rolling_apply(price['Close'],  window, slopee)
    aa = (aa -(window + 1)/2.0 * pd.rolling_sum(price['Close'],window) ) /tmp
    return aa*75/stdev_returns#, intercept

        
def opening_gap(price,window=90,entryZ=0.1,bias=1):
    entryZscore=entryZ
    stdret = pd.rolling_std(price.Close.pct_change(), window=window).shift(1)
    longs = price.Open >= price.Close.shift(1)*(1+entryZscore*stdret)
    shorts = price.Open <= price.Close.shift(1)*(1-entryZscore*stdret)
    price['Forecast'] = 0
    price.loc[longs,'Forecast'] = 1*bias
    price.loc[shorts,'Forecast'] = -1*bias
    return price
    
def opening_gap_stock(price,window=90,entryZ=0.1,bias=1):
    entryZscore=entryZ
    stdret = pd.rolling_std(price.Close.pct_change(), window=window).shift(1)
    #longs = price.Open >= price.Close.shift(1)*(1+entryZscore*stdret)
    #if price.Close > pd.rolling_mean(price.Close, window=20):
    #longs = price.Open >= price.High.shift(1)*(1+entryZscore*stdret)
    #longs = (price.Open >= price.Close.shift(1)*(1+entryZscore*stdret)) & (price.Close > pd.rolling_mean(price.Close, 250))
    #shorts = price.Open <= price.Close.shift(1)*(1-entryZscore*stdret)
    shorts = (price.Open <= price.Close.shift(1)*(1-entryZscore*stdret)) & (price.Close.shift(1) < pd.rolling_mean(price.Close, 20).shift(1)) & (price.Close.shift(1) > pd.rolling_mean(price.Close, 100).shift(1))
    price['Forecast'] = 0
    #price.loc[longs,'Forecast'] = 1*bias
    price.loc[shorts,'Forecast'] = -1*bias
    return price
    
    