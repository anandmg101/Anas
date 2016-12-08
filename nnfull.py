import numpy as np
import scipy as sp
import TrueGenerators as tg
import nnFunc
import config
import time

#%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  #% 20x20 Input Images of Digits
hidden_layer_size = 25;   #% 25 hidden units
num_labels = 1;          #% 10 labels, from 1 to 10
lmbda = 3                          #% (note that we have mapped "0" to label 10)

while True:
    temps = input('Enter lambda value:')
    if nnFunc.is_number(temps):
        break
    print ("Enter digits, pls")
lmbda = float(temps)

while True:
    temps = input('Enter no of iters:')
    if temps.isdigit():
        break
    print ("Enter digits, pls")
s = int(temps)

#%% Load data here

X, y, input_layer_size = tg.TrueGenerators()

X_train, y_train, X_test, y_test = nnFunc.segregate(X,y)

#%% Randomize InitialWeights


initial_Theta1 = nnFunc.randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = nnFunc.randInitializeWeights(hidden_layer_size, num_labels);

# Unroll parameters
initial_theta = np.hstack([initial_Theta1.ravel(), initial_Theta2.ravel()])

#%% Train using fmincg
args = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lmbda)

#print(cst.nnCostFunc(initial_nn_params,input_layer_size, hidden_layer_size, num_labels, X, y, lmbda))
time.sleep(5.5)
theta = sp.optimize.fmin_cg(nnFunc.computeCost, initial_theta, fprime=nnFunc.gradient, args=args,maxiter=s);

theta1 = np.reshape(theta[0:(hidden_layer_size*(input_layer_size+1))],(hidden_layer_size, (input_layer_size+1)))  #5x4
theta2 = np.reshape(theta[(hidden_layer_size*(input_layer_size+1)):],(num_labels, hidden_layer_size+1))     #3x6

np.savetxt('theta1.csv', theta1, delimiter=',')
np.savetxt('theta2.csv', theta2, delimiter=',')


y_pred = nnFunc.predict(theta1, theta2, X_test)
np.savetxt('y_pred.csv', y_pred, delimiter=',')
np.savetxt('y_test.csv', y_test, delimiter=',')
y_pred[y_pred>=0.5]=1
y_pred[y_pred<0.5]=0
np.savetxt('y_pred.csv', y_pred, delimiter=',')

verify = y_test - y_pred
verify[verify==-1]=1

if len(y_test) == len(y_pred):
    print((1-sum(verify)/len(verify))*100)
else:
    print('Length error')
