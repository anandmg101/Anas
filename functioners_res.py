# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:00:02 2016

@author: agnanamoorthy
"""
import skimage.transform as ST
#import skimage.io as io
import skimage.data
import numpy as np

def MRexer(fn, s):
    M = skimage.data.imread(fn, True)
    M = skimage.transform.resize(M,(s,s))
    return M

def TrueGenerator(s):
    data = np.zeros([2000,s*s])
    for i in range(1,251):
        fn = 'B (' + str(i) + ').jpg'

        MRex = MRexer(fn,s)
        #print ("_________________")
        #print ("")
        data[(i-1)*8+0,:] = MRex.ravel()
        data[(i-1)*8+1,:] = ST.rotate(MRex, 90, order = 3).ravel()
        data[(i-1)*8+2,:] = ST.rotate(MRex, 180, order = 3).ravel()
        data[(i-1)*8+3,:] = ST.rotate(MRex, 270, order = 3).ravel()
        MRexFP = np.fliplr(MRex)
        data[(i-1)*8+4,:] = MRexFP.ravel()
        data[(i-1)*8+5,:] = ST.rotate(MRexFP, 90, order = 3).ravel()
        data[(i-1)*8+6,:] = ST.rotate(MRexFP, 180, order = 3).ravel()
        data[(i-1)*8+7,:] = ST.rotate(MRexFP, 270, order = 3).ravel()
    return data

def FalseGenerator(s):
    data = np.zeros([2000,s*s])
    for i in range(1,251):
        fn = 'A (' + str(i) + ').jpg'

        MRex = MRexer(fn,s)
        #print ("_________________")
        #print ("")
        data[(i-1)*8+0,:] = MRex.ravel()
        data[(i-1)*8+1,:] = ST.rotate(MRex, 90, order = 3).ravel()
        data[(i-1)*8+2,:] = ST.rotate(MRex, 180, order = 3).ravel()
        data[(i-1)*8+3,:] = ST.rotate(MRex, 270, order = 3).ravel()
        MRexFP = np.fliplr(MRex)
        data[(i-1)*8+4,:] = MRexFP.ravel()
        data[(i-1)*8+5,:] = ST.rotate(MRexFP, 90, order = 3).ravel()
        data[(i-1)*8+6,:] = ST.rotate(MRexFP, 180, order = 3).ravel()
        data[(i-1)*8+7,:] = ST.rotate(MRexFP, 270, order = 3).ravel()
    return data
