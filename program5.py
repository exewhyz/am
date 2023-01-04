import numpy as np#numpy is to process arrays np supports numerical calculations
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)#training data row 1 2 and 3
y = np.array(([92], [86], [89]), dtype=float)#labels row 1 2 and 3
#normalize the inputs X and y
X = X/np.amax(X,axis=0) # maximum of X array longitudinally col 1 max is 3 2/3 1/3 3/3 col 2 max is 9 9/9 5/9 6/9
y = y/100
#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))
#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)
#Variable initialization
epoch=5000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = 2 #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer
#weight and bias initialization using randomw function to generate numbers uniformly
#w and b for hidden layers
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))#since only one bias value for a layer
#w and b for output layer
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
#draws a random range of numbers uniformly of dim x*y
for i in range(epoch):
    #Forward Propogation
    hinp1=np.dot(X,wh)#dot--> matrix mult
    hinp=hinp1 + bh
    #activation function
    hlayer_act = sigmoid(hinp)
    outinp1=np.dot(hlayer_act,wout)
    outinp= outinp1+ bout
    #activation function
    output = sigmoid(outinp)
    #Backpropagation
    #Formula 1
    EO = y-output
    outgrad = derivatives_sigmoid(output)
    #prod of error at output layer and derivative of sigmoid id the error at a layer Err=O(1-O)
    d_output = EO* outgrad
    #formula 2
    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(hlayer_act)#how much hidden layer wts contributed to error
    #Err=O(1-O)summation*ErrK*Wk 
    d_hiddenlayer = EH * hiddengrad
    #wij usind delta-wij formula
    wout += hlayer_act.T.dot(d_output) *lr# dotproduct of nextlayererror and currentlayerop
   # bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
    #bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
print("Input: \n" + str(X)) 
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)