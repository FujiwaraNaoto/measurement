# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 23:27:20 2022

@author: titan
もともとはpython2系でかかれてるな
https://techacademy.jp/magazine/28381
https://yamaguchiyuto.hatenablog.com/entry/2014/04/28/095451



"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def count(img, x, y, b):
    """ img: 対象画像、x,y: 画像のサイズ、b: ボックスのサイズ """
    i = 0
    j = 0
    c = 0
    while i < x and j < y:
        flag = False
        for k in range(0, b):
            for l in range(0, b):
                if i+k < x and j+l < y:
                    if img.getpixel((i+k,j+l)) == 0:
                        """ ボックスに図形が含まれていたらカウントして次の図形へ """
                        c += 1
                        flag = True
                        break
            if flag:
                break
        i += b
        if i >= x:
            """ ボックスが右端に達したら左端に戻す """
            i = 0
            j += b

    return c # 図形が含まれていたボックスの数を返す



if __name__ == '__main__' :
    filename = 'sample_wind2.PNG'
    #filename = 'sample.png'
    img = Image.open(filename).convert('1') # 画像を読み込んで二値化
    #img.show()グレースケールを表示するだけ
    (x, y) = img.size # 画像サイズを取得
    
    value_x=[]
    value_y=[]
    i = x/10
    while i > x/1000:
        n = count(img, x, y, int(i))
        print(np.log10(int(i)), np.log10(n*i))
        print("i=",int(i), n*i)
        #value_x.append(math.log(i))
        #value_y.append(math.log(n))
        value_x.append(i)
        value_y.append(n*i)
        
        i = int(i / 2)
        
    value_x = np.array(value_x)
    value_y = np.array(value_y)
    value_x_log10 = np.log10(value_x)
    value_y_log10 = np.log10(value_y)
    """
    点もつけるべし
    """
    plt.plot(value_x_log10 , value_y_log10 )
    plt.show()
    
    
    
    
    
    