# -*- coding: utf-8 -*-


#@author: Jeremy Ross


import matplotlib.pyplot as plt

destination ='C:/Users/Jeremy Ross/Documents/Big Data/BigData-TA-3-master/'
branches =['TA3_Cost_Private.txt','TA3_Cost_Public.txt']
Names = ['Cost - Private','Cost - Public']

for x in range(len(branches)):
    txt= open(destination+branches[x])
    txt2=txt.readlines()
    
    df=[x.replace('(','').replace(')','').replace('\n','').split(',') for x in txt2]
    df2 = [[float(i) for i in x]for x in df]
    dfx = [x[0] for x in df2]
    dfy = [x[1] for x in df2]
    hlinex = [x+1 for x in range(100)]
    hliney = [sum(dfy)/len(dfy) for x in range (100)]
    vlinex = [sum(dfx)/len(dfx) for x in range (150)]
    vliney = [x+1 for x in range (150)]
    

    plt.scatter(dfx,dfy)
    plt.plot(hlinex,hliney, color = 'red')
    plt.plot(vlinex,vliney, color = 'red')
    #plt.xlim(min(dfx),max(dfx))
    #plt.ylim(min(dfy),max(dfy))
   
    plt.xlim(40,100)
    plt.ylim(40,100)
    plt.ylabel(ylabel='Predicted Value')
    plt.xlabel(xlabel='Actual Value')
    plt.title(Names[x])
    plt.show()
    
    txt.close()