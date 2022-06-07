# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 00:23:51 2022

@author: titan
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
            all_data.append(data)
            c=data.split(',')
            #print("date=",data,c[1])
            #print(line,c[1])
            
            if(c[1]==''):c[1]=pre #データが空の場合は直前のデータで埋めることにする
            pre=c[1]
            raw_wind_data.append(float(c[1]))
            
            
    data_sz = int(line)
    time = np.arange(0,data_sz+1)
    raw_wind_data = np.array(raw_wind_data)
    dtrend_data = signal.detrend(raw_wind_data)#trendをのける
    value_y = dtrend_data - np.mean(dtrend_data)
    
    
    plot_y = value_y[0:24*31]
    plot_x = time[0:24*31]
    #軸ラベル等を消す方法
    #https://www.delftstack.com/ja/howto/matplotlib/how-to-hide-axis-text-ticks-and-or-tick-labels-in-matplotlib/
    ax = plt.gca()
    plt.plot(plot_x,plot_y)
    plt.xlim([0,plot_x.size-1])
    #ax.tick_params(bottom=False,left=False,right=False,top=False)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig("sample_wind.png")
    plt.savefig("sample_wind.pdf")
    plt.savefig("sample_wind.eps")
    plt.show()
    
    #画像を淵で切り取る処理
    im = Image.open('sample_wind.png')
    (pixel_x,pixel_y)=im.size
    #left upper right lower
    im_crop = im.crop((0+55,0+40, pixel_x-45, pixel_y-40))
    im_crop.save('sample_wind2.png', quality=95)
    
    
    #パワースペクトルを求める処理
    freq,amp_data,fft_data = FFT(value_y,1/(60*60))
    plt.plot(freq,amp_data)
    plt.show()
