#%% -*- coding: utf-8 -*-
import numpy as np
from scipy.special import expit as sigmoid

#%%
def nnCostFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    
    Theta1 = nn_params[0:(input_layer_size+1)*hidden_layer_size]
    Theta1 = Theta1.reshape((hidden_layer_size,input_layer_size+1),order='F')
    Theta2 = nn_params[(input_layer_size+1)*hidden_layer_size:]
    Theta2 = Theta2.reshape((num_labels,hidden_layer_size+1),order='F')

    
    m = size(X, 0);

    J = 0;
    #Theta1_grad = np.zeros(size(Theta1));
    #Theta2_grad = np.zeros(size(Theta2));
    
    eye_matrix = np.eye(num_labels)
    if num_labels == 1:
        y_matrix = y.astype(int)
    else:
        y_matrix = eye_matrix[y-1,:]
    y_matrix.shape = (size(y,0),num_labels)
    
#--------------------------------------------------------------------------
#forward propagation
#--------------------------------------------------------------------------    

    X = np.concatenate((np.ones(((size(X,0)),1)), X),1)
    hidden = sigmoid(X@Theta1.T)
    hidden = np.concatenate((np.ones(((size(hidden,0)),1)), hidden),1)
    
    prediction = sigmoid(hidden@Theta2.T)
    
    for i in range(num_labels):
        J = J + (1/m)*(((np.log(prediction[:,i]).T)@(-y_matrix[:,i])) - (np.log(1-prediction[:,i]).T)@(1-y_matrix[:,i]))
    
    for i in range(1,size(Theta1,1)):
        J= J + lmbda/(2*m)*np.sum(np.square(Theta1[:,i]))
        
    for i in range(1,size(Theta2,1)):
        J= J + lmbda/(2*m)*np.sum(np.square(Theta2[:,i]))
    
    J = np.sum(np.sum(J))
    print (J)
    return J
#--------------------------------------------------------------------------
#backward propagation
#--------------------------------------------------------------------------
"""    D3 = prediction - y_matrix
    Delta2 = D3.T@hidden
    
    Theta2[:,0] = 0
    Theta2_grad = (1/m)*Delta2 + (lmbda/m)*Theta2

    D2 = D3@Theta2[:,1:size(Theta2,1)]*sigmoidGradient(X@Theta1.T)
    
    Theta1[:,0] = 0

    Delta1 = D2.T@X
    Theta1_grad = (1/m)*Delta1 + (lmbda/m)*Theta1 
    print(Theta1_grad)
    print(Theta2_grad)

    grad = np.concatenate((Theta1_grad.ravel(order='F'),Theta2_grad.ravel(order='F')),0)
"""    
    
    
#%%
def size(X,dim = -1):
    #if type(X) == numpy.ndarray:
    if dim == 0:
        return X.shape[0]
    elif dim == 1:
        return X.shape[1]
    else:
        return X.shape
    
#%%
def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))
