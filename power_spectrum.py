
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 00:23:51 2022

@author: titan

submit用
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from PIL import Image


def FFT(corrected_data,dt):
    '''すでにtrend biasがのぞかれたデータが来るとする'''
    #corrected_data_acf = copy.deepcopy(corrected_data) 
    N= corrected_data.shape[0]
    #N = self.corrected_tidal_data_np.shape[0]#サンプル数
    # 高速フーリエ変換　Descrete fast fourier transfer 
    F = np.fft.fft(corrected_data)
    #F = FFT(tidal_data_np) #行けたわ
    # 振幅スペクトルを計算
    Amp = np.abs(F)/np.sqrt(N) #√Nで割っておく

    # パワースペクトルの計算（振幅スペクトルの二乗）
    Pow = Amp ** 2/N#Nで割っておく

    #freq = np.linspace(0, 1.0/dt, N)
    #omega = np.array(range(0,N,1))#ここ,どうすれば良いのか
    freq = np.linspace(0, 1.0/dt, N)#0~(1/dt)をN分割 [Hz] 
        
    return freq,Amp,Pow



if __name__ == '__main__' :
    
    raw_wind_data=[]
    all_data=[]
    file_name= "windspeed2.csv"
    pre='0'
    with open(file_name) as tf:
        for line,data in enumerate(tf):
            #if(line%10==0):
            all_data.append(data)
            c=data.split(',')
            raw_wind_data.append(float(c[1]))
                
            
    
    data_sz = int(line)
    time = np.arange(0,data_sz+1)
    raw_wind_data = np.array(raw_wind_data)
    dtrend_data = signal.detrend(raw_wind_data)#trendをのける
    value_y = dtrend_data - np.mean(dtrend_data)#biasのける
    
    
    plot_y = value_y[0:24*31]
    plot_x = time[0:24*31]

    #パワースペクトルを求める処理
    freq,amp_data,pow_data = FFT(value_y,1/(60*60))
    
    
    #表示
    # log10の場合
    plot_freq_log10 = np.log10(freq)
    plot_pow_log10 = np.log10(pow_data)
    
    """
    plt.plot(plot_freq_log10,plot_pow_log10)
    plt.xlim([0,np.max(plot_freq_log10)])
    plt.ylabel("$log_{10}G[-]$")
    plt.xlabel("$log_{10}f[Hz]$")
    plt.savefig("original_power_spectrum1.png")
    plt.show()
    
    
    """
    
    plot_freq_log10=plot_freq_log10[1:plot_freq_log10.size-1]
    plot_pow_log10 = plot_pow_log10[1:plot_pow_log10.size-1]
    
    plot_freq_log10=plot_freq_log10[1:int(plot_freq_log10.size/3)]
    plot_pow_log10 = plot_pow_log10[1:int(plot_pow_log10.size/3)]
    
    #plot_freq_log10=plot_freq_log10[2*500:plot_freq_log10.size-2*500]
    #plot_pow_log10 = plot_pow_log10[2*500:plot_pow_log10.size-2*500]
    
    print("minx=",np.min(plot_freq_log10),"min_f = ",np.power(10,np.min(plot_freq_log10)) )
    print("maxx=",np.max(plot_freq_log10),"max_f = ",np.power(10,np.max(plot_freq_log10)) )
    
    
    
    slope, intercept = np.polyfit(plot_freq_log10, plot_pow_log10,1)
    print("slope=",slope,"intercept=",intercept)
    line_x = np.arange(np.max([0,np.min(plot_freq_log10)]),np.max(plot_freq_log10), 0.01)
    #line_x = np.arange(0,np.max(plot_freq_log10), 0.01)
    line_y = slope*line_x+intercept
    
    #plt.xlim( [np.max(0.0,np.min(plot_freq_log10)), np.max(plot_freq_log10)])
    #plt.xlim( [np.min(l, np.max(line_x)])
    plt.xlim(np.max([0,np.min(plot_freq_log10)]), np.max(plot_freq_log10))
    
    plt.plot(plot_freq_log10,plot_pow_log10,label="f-G")
    plt.plot(line_x,line_y,label="line")
    plt.ylabel("$log_{10}G[-]$")
    plt.xlabel("$log_{10}f[Hz]$")
    plt.legend()
    plt.savefig("power_spectrum1_and_line.png")
    plt.show()
    #"""
    beta = (-1)*slope
    E = 1
    D = E+(3-beta)/2
    print("D=",D)
    
    """
    #対数グラフでの表示
    plot_freq =freq
    plot_pow = pow_data
    plt.plot(plot_freq,plot_pow)
    #plt.xlim([np.min(plot_freq),np.max(plot_freq)])
    plt.xlim([1,np.max(plot_freq)])
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("$Power[-]$")
    plt.xlabel("$Frequency[Hz]$")
    plt.savefig("original_power_spectrum2.png")
    plt.show()
    """