# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:45:16 2016

@author: agnanamoorthy
"""
#%%
    #import skimage.transform as ST
    #import skimage.io as io
    #import skimage.data
def TrueGenerators():
    import numpy as np
    import functioners

    while True:
        temps = input('Enter pixel values:')
        if temps.isdigit():
            break
        print ("Enter digits, pls")
    s = int(temps)
    XDataTrue = functioners.TrueGenerator(s)
    print('Done generating X True data')
    #input('Press enter to continue:')

    XDataFalse = functioners.FalseGenerator(s)
    print('Done generating X False data')
    #input('Press enter to continue:')

    #Add Y values
    FDataTrue = np.concatenate((XDataTrue,np.ones((np.shape(XDataTrue)[0],1))),1)
    FDataFalse = np.concatenate((XDataFalse,np.zeros((np.shape(XDataFalse)[0],1))),1)

    #Add both True & False
    FData = np.concatenate((FDataTrue,FDataFalse),0)
    #print(np.shape(FData))
    #np.savetxt('test2.csv', FData, delimiter=',')
    """while True:
        value = input("Which Row?:")
        if value == '':
            break
        if int(value)>=0 and int(value)<=3999:
            print(FData[int(value),s*s])"""

    np.random.shuffle(FData)
    print('Data shuffled')

    X = FData[:,:-1]
    y = FData[:,FData.shape[1]-1]
    y.shape = (FData.shape[0],1)
    return X, y, s*s

#%%
    #MRex = np.concatenate((np.zeros([14,80]), M, np.zeros([13,80])))
    #ST.rotate(MRex,270,order = 3)
    #np.shape(N)
