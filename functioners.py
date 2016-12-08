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
    N = skimage.data.imread(fn, True)
    (R, C) = np.shape(N)
    #print ("File:", fn, "Size:", R,C)
    if R > C:
        Cx = int(s/np.shape(N)[0]*np.shape(N)[1])
        #print("Cx:", Cx)
        M = skimage.transform.resize(N,(s,Cx))
        #print ("Size of M:", np.shape(M))
        if Cx%2 == 0:
            Cst = (s - Cx)/2
            Csb = (s - Cx)/2
        else:
            Cst = (s - Cx)/2 + 0.5
            Csb = (s - Cx)/2 - 0.5
        #print ("Spacer size:", Cst, Csb)            
        spaceT = np.zeros([s,int(Cst)])
        spaceB = np.zeros([s,int(Csb)])
        for j in range(int(Cst)):
            spaceT[:,j]=M[:,0]
            if j < int(Csb):
                spaceB[:,j]=M[:,0]
        MRex = np.concatenate((spaceT, M, spaceB),1)
        #print ("Size of MRex:", np.shape(MRex))
    else:
        Cx = int(s/np.shape(N)[1]*np.shape(N)[0])
        #print("Cx:", Cx)
        M = skimage.transform.resize(N,(Cx,s))
        #print ("Size of M:", np.shape(M))
        if Cx%2 == 0:
            Cst = (s - Cx)/2
            Csb = (s - Cx)/2
        else:
            Cst = (s - Cx)/2 + 0.5
            Csb = (s - Cx)/2 - 0.5          
        #print ("Spacer size:", Cst, Csb)
        spaceT = np.zeros([int(Cst),s])
        spaceB = np.zeros([int(Csb),s])
        for j in range(int(Cst)):
            spaceT[j,:]=M[0,:]
            if j < int(Csb):
                spaceB[j,:]=M[0,:]
        MRex = np.concatenate((spaceT, M, spaceB),0)
        #print ("Size of MRex:", np.shape(MRex))
    return MRex
    
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
