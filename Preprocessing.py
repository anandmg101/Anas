# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:45:16 2016

@author: agnanamoorthy
"""
#%%
import skimage.transform as ST
#import skimage.io as io
#import skimage.data
import numpy as np
from functioners import MRexer

data = np.zeros([2000,6400])
for i in range(1,251):
    if i<10:
        fn = str(0) + str(0) +str(i) + '.jpg'
    if i>=10 and i<100:
        fn = str(0) + str(i) + '.jpg'
    if i>100:
        fn = str(i) + '.jpg'
    print (fn)
    MRex = MRexer(fn)
    print ("_________________")
    print ("")
    data[(i-1)*8+0,:] = MRex.ravel()
    data[(i-1)*8+1,:] = ST.rotate(MRex, 90, order = 3).ravel()
    data[(i-1)*8+2,:] = ST.rotate(MRex, 180, order = 3).ravel()
    data[(i-1)*8+3,:] = ST.rotate(MRex, 270, order = 3).ravel()
    MRexFP = np.fliplr(MRex)
    data[(i-1)*8+4,:] = MRexFP.ravel()
    data[(i-1)*8+5,:] = ST.rotate(MRexFP, 90, order = 3).ravel()
    data[(i-1)*8+6,:] = ST.rotate(MRexFP, 180, order = 3).ravel()
    data[(i-1)*8+7,:] = ST.rotate(MRexFP, 270, order = 3).ravel()  

    
#%%
    #MRex = np.concatenate((np.zeros([14,80]), M, np.zeros([13,80])))
    #ST.rotate(MRex,270,order = 3)
    #np.shape(N)