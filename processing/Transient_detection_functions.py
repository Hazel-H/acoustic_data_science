# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:40:22 2021

@author: -
"""

import numpy as np 
import pandas as pd


def remove_nans(df):
    '''
    Removes any rows containing nan values and two rows either side of each of 
    these rows, where data is anomalous.
    '''
    m = df.isna().any(axis=1)
    return df[~(m | m.shift(fill_value=False) | m.shift(-1, fill_value=False) | m.shift(-2, fill_value=False))]


def broadband_SPL_calc(df): 
    'Function that takes in dataframe (containing TOL PAMGuide csv output with time column dropped) and returns'
    'dataframe of the broadband SPL and background for each time interval'
    
    #function to map onto dataframe 
    def broadband_func(x): 
        return 10**(x/10)
    
    #apply function, map to dataframe, sum by row, take the log and then normalise to maximum value of zero 
    SPL = df.applymap(broadband_func)
    SPL_sum = SPL.sum(axis=1)
    broadband_SPL = 10*np.log10(SPL_sum)
    broadband_SPL_normalised = broadband_SPL - np.max(broadband_SPL)
    
    #create dataframe to present results 
    SPL = pd.DataFrame({'broadband_SPL':broadband_SPL_normalised})
    
    #calculate 'background' sound level using moving average
    window = 5*60*2 
    SPL['background' ] = SPL['broadband_SPL'].rolling(window).mean()    
    
    return SPL


def loud_noises(df): 
    'Function takes in broadband SPL dataframe and adds label True/False depending if time interval is classified as loud or not'
    'returns original dataframe with loud label and dataframe with just loud noises'
    
    # detection of transient events - example if mean noise of 'crack' is 10% louder than mean background noise 
    df['loud'] = np.where(df['broadband_SPL']>(df['background']+15), True, False)

    #new dataframe of transient events for plotting 
    loud_noise = df[df['loud']==True]
    loud_noise.reset_index(inplace=True)
    
    #prints number of loud noise events as number and percentage of overall time 
    print('Number of loud noise events detected {}, {:.2}% of total time'.format(len(loud_noise), len(loud_noise)/len(df)*100))

    return df, loud_noise


def transients_from_loud(df):
    'FASTER METHOD - inserts new column (transient) with a True / False flag for a transient event '
    
    df['trans_shft_down'] = np.concatenate((np.array([True]), df['loud'][:-1]))
    df['trans_shft_up'] = np.concatenate((df['loud'][1:], np.array([True])))
    df['transient'] = df['loud'] & (df['trans_shft_down']==False) & (df['trans_shft_up']==False)
    df.drop(columns=['trans_shft_down', 'trans_shft_up'], inplace=True)
    
    return df



