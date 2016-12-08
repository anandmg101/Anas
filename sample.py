import numpy as np
import scipy.optimize as sp

weights = np.array([150.1,50.1,100.1])

for i in range(10):
    X = np.array([2,5,3])
    y = sum(X*weights)
    weights = weights + 1/50*X*[850-y]
    if abs(850-y)*100000000 <= 1: continue
    print (y)


final_price = sum(weights*X)
print(weights)
print(final_price)
