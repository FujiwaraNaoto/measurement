# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Thu May 19 01:11:36 2022
一応完成

@author: titan
https://www.data.jma.go.jp/kaiyou/data/db/tide/genbo/2022/202202/hry202202TK.txt
線形回帰と　0 biasも行う必要がある 

ACFについて https://momonoki2017.blogspot.com/2018/03/python7.html

トレンドについて　https://data.gunosy.io/entry/statsmodel_trend

https://org-technology.com/posts/detrend.html

matplotlibについてはグラフ表記について
https://qiita.com/kakiuchis/items/798c00f54c9151ab2e8b
https://qiita.com/qsnsr123/items/325d21621cfe9e553c17
https://villageofsound.hatenadiary.jp/entry/2014/11/06/155824

相互相関
https://qiita.com/inoory/items/3ea2d447f6f1e8c40ffa

"""

    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ptick #べき乗表記に必要
import statsmodels.api as sm #ACFに必要
import pandas as pd
import copy
from scipy import signal
from scipy import misc
#from scipy import signal.correlate
from sklearn.linear_model import LinearRegression
#from statsmodels.tsa.seasonal import seasonal_decompose


"""
#O(N^2) 遅い
def FFT(array):

    n = array.shape[0]
    DFT = np.array([0]*n)#この書き方できるのか
    
    for k in range(0,n-1):
        
        for num in range(0,n-1):
            DFT[k] += array[num]*np.exp(-1j*(2*np.pi*(k+1)*num)/N)
    
    return DFT
"""


def drawFig(x_value,y_value,title_name='',label_x='x',label_y='y',file_name='tmp.pdf',font_family='Times New Roman',font_size = 17,is_show=False,axis_option=True):
    '''グラフをかくためだけの関数'''
    fig = plt.figure()
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size
    #plt.subplot(121)
    if(title_name==''):
        pass
    else:
        plt.title(label=title_name,loc='center')
    
    plt.plot(x_value, y_value)
    plt.xlabel(label_x, fontsize=15)
    plt.ylabel(label_y, fontsize=15)
    plt.grid()
    
    
    plt.xlim(x_value[0], x_value[-1])#-1で後ろのデータ
    
    if(axis_option):
        plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) #こいつ！！
        #plt.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
        plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0)) #こいつ！！
        #plt.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
        #plt.xticks( np.arange(x_value[0] ,x_value[-1],(x_value[-1]-x_value[0])/3 ) )
        #plt.gca().axis.set_major_formatter(plt.FormatStrFormatter('%.3f'))#y軸小数点以下3桁表示
        plt.gca().xaxis.get_major_formatter().set_useOffset(False)
        
    
    #plt.ylim(Yの最小値, Yの最大値)
    
    #leg = plt.legend(loc=1, fontsize=25)
    #leg = plt.legend(loc='best')
    #leg = plt.legend(loc='north east outside')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)

    #leg.get_frame().set_alpha(1)
    if(is_show):
        plt.show()
    plt.savefig(file_name)
    
    return fig


class TideAnalysis:
    
    
    
    def __init__(self,open_file_name,dt):
        '''データを読み込むためのもの'''
        
        self.file_name=open_file_name
        self.raw_tidal_data=[]#潮のデータを格納
        self.raw_tidal_data_np=np.array([])#numpyがたの配列
        
        
        self.max_tidal_data=[]#満潮の時
        self.min_tidal_data=[]#干潮の時
        
        self.dt = dt
        
        
    
        with open(self.file_name) as tf:#一行ずつ処理していく
           
            for idx,line in enumerate(tf):
            #print(idx,line)
                for i in range(0,72,3):# 0~ 69 まで+3 72を超えない [0,72)
                    self.raw_tidal_data.append(float(line[i:i+3]))
                    #print("i=",i)
                
                i+=3#scopeの関係上仕方ないのである
                self.year = line[i:i+2]
                #print("year=",year)
                i+=2
                self.month = line[i:i+2]
                #print("month=",month)
                i+=2
                self.day = line[i:i+2];
                i+=2
                self.place = line[i:i+2]
                i+=2
                #print("year=",year,"day=",day,"month=",month,"place=",place)
                
                #満潮のとき
                for j in range(0,4,1):
                    hour = int(line[i:i+2])
                    i+=2
                    minute = int(line[i:i+2])
                    i+=2
                    height = float(line[i:i+3])
                    i+=3
                    #print("hour=",hour,"minute=",minute,"height=",height)
                    if(hour==99 and minute==99 and height==999.0):
                        pass
                    else:
                        self.max_tidal_data.append((hour,minute,height))
                
                for j in range(0,4,1):
                    hour = int(line[i:i+2])
                    i+=2
                    minute = int(line[i:i+2])
                    i+=2
                    height = float(line[i:i+3])
                    i+=3
                    
                    if(hour==99 and minute==99 and height==999.0):
                        #print("*")
                        pass
                    else:
                        self.min_tidal_data.append((hour,minute,height))
                       
            #データ読み込み部分終わり        
            
            self.raw_tidal_data_np=np.array(self.raw_tidal_data)
            self.sample_num = self.raw_tidal_data_np.shape[0]
        
    
   
    

    def Trending_and_bias(self):
        """bias と trendを除く"""
        
        #self.raw_tidal_data_np=np.array(self.raw_tidal_data)
        tidal_data_removed_trend = signal.detrend(self.raw_tidal_data_np)#trendをのける
        
        #0バイアスにする.平均とってその分をのけるだけ
        mean_value = np.sum(tidal_data_removed_trend)/self.sample_num
        tidal_data_removed_trend_and_bias = tidal_data_removed_trend - np.mean(tidal_data_removed_trend) #これでtidal_data_npの全要素から mean_valueぶんがひかれることになる
        
        #fig0 = drawFig(time,tidal_data_removed_trend_and_bias,title_name='tidal data',label_x='time[hour]',label_y='tidal data',file_name="tidaldata.png",is_show=True)
        #fig0 = drawFig(time,tidal_data_removed_trend_and_bias,label_x='time[hour]',label_y='tidal data',file_name="tidaldata.png",is_show=True)
        self.corrected_tidal_data_np = tidal_data_removed_trend_and_bias
        return tidal_data_removed_trend_and_bias
    """
    def FFT(self,array):

        n = array.shape[0]
        DFT = np.array([0]*n)#この書き方できるのか
        
        for k in range(0,n-1):
            for num in range(0,n-1):
                DFT[k] += array[num]*np.exp(-1j*(2*np.pi*(k+1)*num)/N)
        
        return DFT
    """
    
    def FFT(self):
        
        #if(self.corrected_tidal_data_np.shape[0]==0):
        tidal_data_removed_trend_and_bias = self.Trending_and_bias()
        self.corrected_tidal_data_np = copy.deepcopy(tidal_data_removed_trend_and_bias)
        
        
        N = self.sample_num
        #N = self.corrected_tidal_data_np.shape[0]#サンプル数

        # 高速フーリエ変換　Descrete fast fourier transfer 
        F = np.fft.fft(self.corrected_tidal_data_np)
        #F = FFT(tidal_data_np) #行けたわ
        # 振幅スペクトルを計算
        Amp = np.abs(F)/np.sqrt(N) #√Nで割っておく

        # パワースペクトルの計算（振幅スペクトルの二乗）
        Pow = Amp ** 2/N#Nで割っておく

        #freq = np.linspace(0, 1.0/dt, N)
        #omega = np.array(range(0,N,1))#ここ,どうすれば良いのか
        freq = np.linspace(0, 1.0/self.dt, N)#0~(1/dt)をN分割 [Hz] 
        
        return freq,Amp,Pow
    
    def ACF(self,lags=40):
        
        #if(self.corrected_tidal_data_np.shape[0]==0):
        tidal_data_removed_trend_and_bias = self.Trending_and_bias()
        self.corrected_tidal_data_np = copy.deepcopy(tidal_data_removed_trend_and_bias)
        
        
        tidal_acf_y = sm.tsa.stattools.acf(self.corrected_tidal_data_np,nlags=lags)
        n = tidal_acf_y.shape[0]
        tidal_acf_x = np.arange(0,n*self.dt,self.dt)
        
        #print("tidal_acf_x =",tidal_acf_x,"tidal_acf_y =",tidal_acf_y )
        return tidal_acf_x,tidal_acf_y
        
        
if __name__=='__main__':
    
    dt=3600
    #松山
    file_name1 = "hry202203MT.txt"
    tide1=TideAnalysis(file_name1,dt=3600)
    freq1,Amp1,Pow1 = tide1.FFT()
    
    
    fig1_1 = drawFig(freq1,Amp1,title_name='MT202203 Amplitude',label_x='f[Hz]',label_y='Amplitude',file_name="FFT.png",is_show=True)
    fig1_2 = drawFig(freq1,Pow1,title_name='MT202203 Power',label_x='f[Hz]',label_y='Power',file_name="POW.png",is_show=True)
    
    #自己相関
    tidal_acf_x1,tidal_acf_y1 = tide1.ACF(lags=40) 
    fig1_3 = drawFig(tidal_acf_x1/3600,tidal_acf_y1,title_name='MT202203 ACF',label_x='time[hour]',label_y='r',file_name="rx.png",is_show=True)
    
    
    
    
    
    #高知
    file_name2 = "hry202203KC.txt"
    tide2=TideAnalysis(file_name2,dt=3600)
    freq2,Amp2,Pow2 = tide2.FFT()
    
    
    fig2_1 = drawFig(freq2,Amp2,title_name='KC202203 Amplitude',label_x='f[Hz]',label_y='Amplitude',file_name="FFT.png",is_show=True)
    fig2_2 = drawFig(freq2,Pow2,title_name='KC202203 Power',label_x='f[Hz]',label_y='pow',file_name="POW.png",is_show=True)
    
    #自己相関
    tidal_acf_x2,tidal_acf_y2 = tide2.ACF(lags=40) 
    fig2_3 = drawFig(tidal_acf_x2/3600,tidal_acf_y2,title_name='AC202203 ACF',label_x='time[hour]',label_y='r',file_name="ry.png",is_show=True,axis_option=False)
    
    
    # https://morioh.com/p/9bf6d605cec4
    #相互相関
    
    Rx0 = np.std(tide1.corrected_tidal_data_np)#標準偏差
    Ry0 = np.std(tide2.corrected_tidal_data_np)
    
    #mode=valid or full
    corr = np.correlate(tide1.corrected_tidal_data_np/Rx0, tide2.corrected_tidal_data_np/Ry0,mode="full")
    
    #入力ベクトルの大きさで規格化 https://blog.colorfulwires.jp/entry/normalized_cross_correlation
    # tide1.corrected_tidal_data_np.shape[0]= tide2.corrected_tidal_data_np.shape[0]
    corr_y = corr[0:300]/tide1.corrected_tidal_data_np.shape[0]
    
    time = np.arange(0,corr_y.shape[0]*dt,dt)/3600#hourにする
    
    fig = drawFig(time,corr_y,title_name='MT and KC CCF',label_x='time[hour]',label_y='correlation',file_name="FFT.png",is_show=True)
    
    
    
    pp = PdfPages('Save_answer_PDF.pdf')
    pp.savefig(fig1_1)
    pp.savefig(fig1_2)
    pp.savefig(fig1_3)
    pp.savefig(fig)
    pp.close()
    
    
    
    