# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:25:36 2020

@author: Christopher
"""

import matplotlib.pyplot as plt

destination = 'C:/Users/Christopher/Downloads/BigData-TA-3-master/'
branches = ['TA3_SF_Public.txt','TA3_SF_Private.txt']
names = ['SF - Public','SF - Private']

for x in range(len(branches)):
    
    txt = open(destination+branches[x])
    txt_2 = txt.readlines()
    
    df = [x.replace('(','').replace(')','').replace('\n','').split(',') for x in txt_2]
    df_2 = [[float(i) for i in x] for x in df]
    df_x = [x[0] for x in df_2]
    df_y = [x[1] for x in df_2]
    h_line_x = [x+1 for x in range(100)]
    h_line_y = [sum(df_y)/len(df_y) for x in range(100)]
    v_line_x = [sum(df_x)/len(df_x) for x in range(100)]
    v_line_y = [x+1 for x in range(100)]
    diff = [x[0]-x[1] for x in df_2]
    
    plt.scatter(df_x,df_y)
    plt.plot(h_line_x,h_line_y,color='red')
    plt.plot(v_line_x,v_line_y,color='red')
    plt.xlim(min(df_x),max(df_x))
    plt.ylim(min(df_y),max(df_y))
    plt.xlim(40,100)
    plt.ylim(40,100)
    plt.ylabel(ylabel='Actual Value')
    plt.xlabel(xlabel='Predicted Value')
    #plt.qca().set_aspect('equal')
    plt.title(names[x])
    plt.show()
    
    txt.close()