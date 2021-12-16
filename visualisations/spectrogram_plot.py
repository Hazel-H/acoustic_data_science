# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 13:59:44 2021
@author: hjh44 

Function to plot spectrogram over timeperiod of given pandas dataframe

"""
import numpy as np 
import matplotlib.pyplot as plt

def spectrogram_plot(df): 
    #create a string with the start time
    start_time = df['timestamp'].iloc[0]
    end_time = df['timestamp'].iloc[-1]
    start_string = start_time.strftime('%H:%M:%S')
    end_string = end_time.strftime('%H:%M:%S on %B the %dth, %Y')

    #drop columns
    df.drop(['unnormalised_broadband_spl', 'broadband_spl', 'background_spl', 'loud', 'loud', 'filename', 'timestamp', 'short_transient' ], axis=1, inplace=True)
    
    #reset index - index relates to half second time intervals 
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    
    #time axis - time in minutes, divide by 60 * 2 (factor of 2 as each interval is half a second)
    time = df.index.to_numpy()
    time = time.astype(np.float)
    time = time/(60*2)
    
    #frequency axis as float
    freq = df.columns.to_numpy()
    freq = freq.astype(np.float)
    
    #plotting
    plt.figure(figsize=(16, 8))
    plt.pcolormesh(time, freq, df.transpose(), cmap='rainbow', shading='auto')
    plt.yscale('log')
    plt.xlabel('Time (minutes)')
    plt.ylabel('TOL frequencies (Hz)')
    plt.colorbar(label = r'SPL level (dB re. 1 $\mu$ Pa)')
    plt.title('Spectrogram from %s to %s' %(start_string, end_string))