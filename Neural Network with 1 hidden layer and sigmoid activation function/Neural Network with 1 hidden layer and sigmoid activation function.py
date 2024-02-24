import numpy as np
import json
from random import random
import time

class SingleLayerNN(object):

    def __init__(self, X=2, H=5, Y=1, W=None): # X is the number of inputs, H is the number of nodes in the hidden layer, Y is the number of outputs, W is a set of pretrained weights

        self.X = X # Inputs
        self.H = H # Hidden layer nodes
        self.Y = Y # Outputs

        self.L = np.concatenate(([X], [H], [Y])) # Array of the number of outputs from each layer

        if W == None:
            w = []
            w.append(np.random.rand(self.X+1,self.H).tolist()) # Initialses the weights by creating a random value between 0 and 1 for each input to each node in the hidden layer, +1 is bias
            w.append(np.random.rand(self.H+1,self.Y).tolist()) # Initialses the weights by creating a random value between 0 and 1 for each hidden layer node to each output, +1 is bias
            self.W = w
        else:
            self.W = W

        allDeltas = [] # A data structure to store the deltas produced during backpropagation to be used during gradient decent
        allDeltas.append(np.zeros((self.X+1,self.H))) 
        allDeltas.append(np.zeros((self.H+1,self.Y)))
        self.allDeltas = allDeltas

        out = [] # A data structure to store the outputs of the neurons produced when the network is run
        out.append(np.zeros(X).tolist())
        out.append(np.zeros(H).tolist())
        out.append(np.zeros(Y).tolist())
        self.out = out

    def FF(self, input): # Runs the network, input is the array of inputs
        self.out[0] = input # The output of the input layer is just the input
        for k in range(2): # 2 as there is 2 sets of weights, first goes from input to hidden layer, then the second goes from hidden to the output
            input = np.concatenate(([1], input)) # Adds a bias of 1 to the start of the input array
            out = np.zeros(self.L[k+1]).tolist() # Initialses an output array of zero's for this layer
            for j in range(self.L[k+1]): # Loops through the outputs
                for i in range(len(input)): # Loops through the inputs for this layer summing up the inputs multiplied by the weights for each output to this layer
                    out[j] += input[i] * self.W[k][i][j] # Multiplies each input value with the corrisponding weight
                out[j] = self.sigmoid(out[j]) # Sends the total sums through the sigmoid function
            input = out # sets the input to the next layer as the output from this layer
            self.out[k+1] = out # Stores the outputs to be used later for backpropagation
        return out
    
    def train_nn(self, inputs, targets, epochs, learningRate): 
        for _ in range(epochs): # Loops through the epochs
            for i in range(len(inputs)): # Loops through the sets of inputs
                input = inputs[i] # inputs for this training
                target = targets[i] # outputs for this training

                out = self.FF(input) # Runs the network on the test data
                error = out - target # Gets the error from the output
                deltas = self.BP(error) # Runs back propagation
                self.GD(deltas,learningRate) # Runs gradient decent

    def BP(self, errors):
        for j in reversed(range(len(self.L)-1)): # Loops through the sets of weights in reverse
            outputs = self.out[j+1] # Gets the outputs of the layer
            inputs = np.concatenate(([1] ,self.out[j])) # Adds the bias to the inputs to the layer
            layerDeltas = []
            errDerArray = []
            for i in range(len(outputs)): # Loops through the outputs from this layer
                error = errors[i] # Gets the error for this output
                out = outputs[i] # Gets the output for this layer
                errDer = error * self.sigmoid_Der(out) # Error * the derivative of the sigmoid function to get the value used for both gradient decent and to calculate the errors for the previous layers
                                                        # We are giving as input to the sigmoid_Der function the output from the sigmoid function for this layer instead of the input to the sigmoid for this layer
                                                        # because the derrivative of sigmoid can be calculated as the output of sigmoid multiplied by 1-the output of sigmoid.
                delta = errDer * inputs # Multiply by the inputs for this output to get delta that will be used during gradient decent
                layerDeltas.append(delta)
                errDerArray.append(errDer)
            self.allDeltas[j] = layerDeltas

            errors = []
            # This part gets the errors for the previous layer
            for l in range(len(self.W[j])-1): # -1 as we dont need to backpropagate the error from the bias node
                err = 0
                for k in range(len(errDerArray)): # Loops through the array of error*dervitives, one for each output of this layer
                    err += errDerArray[k] * self.W[j][l+1][k] # Sums up the errors for the previous layer from the multiplication between the errors, derivitives and weight values from this layer
                errors.append(err)

        return self.allDeltas

    def GD(self, allDeltas, learningRate): # Runs gradient decent by looping through the layers of weight matrices and adding to each weight the 
                                            # product of the learning rate and the delta calculated for the weight during backpropagation
        for k in range(len(self.L)-1): # Loops through the layers
            deltas = np.array(allDeltas[k]).T # Uses the transpose of the delta matrix for this layer to get it in the right shape
            for i in range(len(deltas)): # Loops through the deltas/weight matrix (They are the same size)
                delta = deltas[i]
                for j in range(len(self.W[k][i])): # Loops through the deltas/weight matrix (They are the same size)
                    self.W[k][i][j] = self.W[k][i][j] - (delta[j] * learningRate) # Adds the product of the learning rate and the delta calculated during backpropagation to the weight

    def sigmoid(self,x): # Logarithmic sigmoid is calculated by 1/(1+(e^-x)) where x is the input to the function. Always returns a value between 0 and 1 exclusive.
        y=1.0/(1+np.exp(-x))
        return y

    def sigmoid_Der(self, x): # Derrivative of the Logarithmic sigmoid function where x is not the input to sigmoid but instead x is the output to sigmoid
        sig_der=x*(1.0-x)
        return sig_der

training_inputs = np.array([[np.random.rand() for _ in range(2)] for _ in range(10000)])   # This creates 10,000 training sets of input pairs where each input is a number between 0 and 1
targets = []
for input1, input2 in training_inputs: # Sets the target output for each pair of inputs as the larger input as the first output and the smaller input as the second output
    if input1 >= input2:
        targets.append([input1,input2])
    else:
        targets.append([input2,input1])
targets = np.array(targets)

nn=SingleLayerNN(2, 5, 2)   #creates a NN with 2 inputs, 2 hidden layers and 1 ouput

#Testing data to identify if Network trained well. 
input = np.array([0.4, 0.6])      # after training this tests the train network 
target = np.array([0.6,0.4])         # for this target value.  

NN_outputB = nn.FF(input) # Gets the output of our test data before training

nn.train_nn(training_inputs, targets, 10, 0.1)  # trains the network with 0.1 learning rate for 10 epochs

NN_outputA = nn.FF(input) # Gets the output of our test data after training

# Writes over the file containing the weights for the network that I trained for 10,000 epochs with 10,000 sets of training data in each epoch
""" epoch10000nnFileWrite = open("singleLayer10000EpochWeights.txt", "w") 
epoch10000nnFileWrite.write(json.dumps(nn.W))
epoch10000nnFileWrite.close() """

epoch10000nnFileRead = open("singleLayer10000EpochWeights.txt", "r")
epoch10000nnWeights = json.loads(epoch10000nnFileRead.read())
nn10000Epoch = SingleLayerNN(2,5,2,epoch10000nnWeights)
NN10000_output = nn10000Epoch.FF(input)

print("=============== Testing the Network Screen Output ===============")
print ("Test input is ", input)
print()
print("Target output is ",target)
print()
print("Neural Network actual output BEFORE TRAING is ",NN_outputB, "there is an error (not MSQE) of ",target-NN_outputB)
print()
print("Neural Network actual output AFTER TRAING is ",NN_outputA, "there is an error (not MSQE) of ",target-NN_outputA)
print()
print("As you can see the nn trained its self and has become far more accurate at sorting the input in descending order")
print()
print("Approx 100 epochs are needed to train for a good accuracy, takes my computer about 50 seconds")
print("Approx 1000 epochs are needed to train for a really good accuracy, takes my computer about 8.3 minutes")
print()
print("Neural Network actual output 10000 epochs training done earlier is ",NN10000_output, "there is an error (not MSQE) of ",target-NN10000_output)
print("=================================================================")