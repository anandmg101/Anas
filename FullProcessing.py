import numpy as np
import scipy as sp
import nnCostFunc as cst
import nnGradFunc as grad
import TrueGenerators as tg

sigmoid = sp.special.expit

#%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  #% 20x20 Input Images of Digits
hidden_layer_size = 25;   #% 25 hidden units
num_labels = 1;          #% 10 labels, from 1 to 10
lmbda = 3                          #% (note that we have mapped "0" to label 10)

                
#%% Load parameter here


X, y, input_layer_size = tg.TrueGenerators()


#%% Randomize InitialWeights

def randInitializeWeights(L_in, L_out, epsilon=0.12):
    W = np.zeros((L_out, 1 + L_in))
    W = (2 * epsilon - epsilon)*np.random.rand(L_out, 1 + L_in)
    return W

    
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

# Unroll parameters
initial_nn_params = np.concatenate((initial_Theta1.ravel(order='F'),initial_Theta2.ravel(order='F')),0)



#%% Train using fmincg
args = (input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)

#print(cst.nnCostFunc(initial_nn_params,input_layer_size, hidden_layer_size, num_labels, X, y, lmbda))
while True:
    temps = input('Enter no of iters:')
    if temps.isdigit():
        break
    print ("Enter digits, pls")
s = int(temps)
nn_params = sp.optimize.fmin_bfgs(cst.nnCostFunc, initial_nn_params, fprime=grad.nnOptimizeFunc, args=args,maxiter=s);


Theta1 = nn_params[0:(input_layer_size+1)*hidden_layer_size]
Theta1 = Theta1.reshape((hidden_layer_size,input_layer_size+1),order='F')
Theta2 = nn_params[(input_layer_size+1)*hidden_layer_size:]
Theta2 = Theta2.reshape((num_labels,hidden_layer_size+1),order='F')

#%%
def predict(Theta1, Theta2, X):
    m = X.shape[0]

    #p = np.zeros(m, 1);
    X = np.concatenate((np.ones((m,1)),X),1)
                        
    h1 = sigmoid(X@Theta1.T)
    n = h1.shape[0]
    h1 = np.concatenate((np.ones((n,1)),h1),1)
    h2 = sigmoid(h1@Theta2.T)
    
    p = np.argmax(h2, axis=1)
    
    return p
#%%
pred = predict(Theta1, Theta2, X);
