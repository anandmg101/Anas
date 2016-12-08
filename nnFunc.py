import numpy as np
import config

#%%
def sigmoid(z):
    return 1 / (1+np.exp(-z))

#%%
def sigmoid_gradient(z):
    return sigmoid(z)*(1-sigmoid(z))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def randInitializeWeights(layer_in, layer_out):
    matrix = np.zeros((layer_out, 1 + layer_in))
    epsilon_init = 0.12
    matrix = np.random.rand(layer_out, 1+layer_in) * 2 * epsilon_init -epsilon_init
    return matrix

def segregate(X,y):
    m = len(X)

    data = np.hstack((X,y))
    np.random.shuffle(data)

    X = data[:,:-1]
    y = data[:,data.shape[1]-1]
    y.shape = (data.shape[0],1)

    train_len = int(m*2/3)

    X_train = X[:train_len,:]
    y_train = y[:train_len,:]
    y_train.shape = (X_train.shape[0],1)

    X_test = X[train_len:,:]
    y_test = y[train_len:,:]
    y_test.shape = (X_test.shape[0],1)

    return X_train, y_train, X_test, y_test

#%%
def gradient(theta, *args):

    input_layer_size, hidden_layer_size, num_labels, X, y, lmbda = args

    m = len(X)

    y_bin = np.zeros((m, num_labels))

    if num_labels==1:
        y_bin = y
    else:
        for i in range(m):
            y_bin[i, y[i]] = 1

    theta1 = np.reshape(theta[0:(hidden_layer_size*(input_layer_size+1))],(hidden_layer_size, (input_layer_size+1)))  #5x4
    theta2 = np.reshape(theta[(hidden_layer_size*(input_layer_size+1)):],(num_labels, hidden_layer_size+1))     #3x6

    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    #forward

    a_1 = np.hstack((np.ones((m, 1)), X))   #5x4

    z_2 = np.dot(a_1, theta1.transpose())   #5x5
    a_2 = sigmoid(z_2)                      #5x5

    a_2 = np.hstack((np.ones((m, 1)), a_2)) #5x6
    z_3 = np.dot(a_2, theta2.transpose())   #5x3

    h = sigmoid(z_3)                        #5x3


    #backward

    delta3 = h - y_bin                      #5x3
    delta2 = np.dot(delta3, theta2[:, 1:hidden_layer_size+1]) * sigmoid_gradient(z_2) #5x5

    D1 = np.dot(delta2.transpose(), a_1)    #5x4
    D2 = np.dot(delta3.transpose(), a_2)    #3x6

    theta1_grad = D1/m      #5x4
    theta2_grad = D2/m      #3x6

    #regularization
    theta1_grad[:, 1:input_layer_size+1] = theta1_grad[:, 1:input_layer_size+1] +lmbda/m*  theta1[:, 1:input_layer_size+1]
    theta2_grad[:, 1:hidden_layer_size+1] = theta2_grad[:, 1:hidden_layer_size+1] +lmbda/m*  theta2[:, 1:hidden_layer_size+1]

    #unroll
    grad = np.hstack([theta1_grad.ravel(), theta2_grad.ravel()])
    return grad


def computeCost(theta, *args):

    input_layer_size, hidden_layer_size, num_labels, X, y, lmbda = args

    m = len(X)

    y_bin = np.zeros((m, num_labels))

    if num_labels==1:
        y_bin = y
    else:
        for i in range(m):
            y_bin[i, y[i]] = 1

    theta1 = np.reshape(theta[0:(hidden_layer_size*(input_layer_size+1))],(hidden_layer_size, (input_layer_size+1))) #5x4
    theta2 = np.reshape(theta[(hidden_layer_size*(input_layer_size+1)):],(num_labels, hidden_layer_size+1)) #3x6

    a_1 = np.hstack((np.ones((m, 1)), X))   #5x4

    z_2 = np.dot(a_1, theta1.transpose())   #5x5
    a_2 = sigmoid(z_2)                      #5x5

    a_2 = np.hstack((np.ones((m, 1)), a_2)) #5x6
    z_3 = np.dot(a_2, theta2.transpose())   #5x3

    h = sigmoid(z_3)

    cost = np.sum(-y_bin * np.log(h)    -    (1-y_bin) * np.log(1-h))/m


    #regularization

    theta1_sq = theta1[:, 1:input_layer_size+1] * theta1[:, 1:input_layer_size+1];
    theta2_sq = theta2[:, 1:hidden_layer_size+1] * theta2[:, 1:hidden_layer_size+1];

    cost = cost + lmbda/(2.0*m)*(np.sum(theta1_sq) + np.sum(theta2_sq))
    config.count += 1
    if config.count%500 == 0:
        print('Completed ', config.count//500*500, ' iterations')    
    return cost

#%%
def predict(Theta1, Theta2, X):
    m = len(X)
    h1 = sigmoid(np.hstack((np.ones((m,1)),X))@Theta1.T)
    h2 = sigmoid(np.hstack((np.ones((m,1)),h1))@Theta2.T)

    return h2

#%%



"""
def computeNumericalGradient(J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 0.0001

    for p in range(1, np.size(theta)):
        perturb[p] = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)

        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    return numgrad

def gradientChecking(lmbda):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    theta2 = randInitializeWeights(hidden_layer_size, num_labels)

    X = np.random.rand(m, input_layer_size)
    y = np.array([1, 2, 0, 1, 2])

    nn_params = np.hstack([theta1.ravel(), theta2.ravel()])

    #calculate gradient with function
    grad = gradient(nn_params, X, y, input_layer_size, hidden_layer_size, num_labels, lmbda)
    #calculate numerical gradient
    num_grad = computeNumericalGradient(lmbdada theta: computeCost(theta, X, y, input_layer_size, hidden_layer_size, num_labels, lmbda), nn_params)

    print('Function Gradient', 'Numerical Gradient')
    for i in range(len(grad)):
        print(grad[i], num_grad[i])

    diff = np.linalg.norm(num_grad-grad)/np.linalg.norm(num_grad+grad)
    print('Relative Difference: ')
    print(diff)

"""
