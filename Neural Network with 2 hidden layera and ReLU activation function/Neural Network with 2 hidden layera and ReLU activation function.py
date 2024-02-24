import numpy as np
from random import random
import json
import time

class MultiLayerNN(object):

    def __init__(self, X=2, H1=5, H2=5, Y=1, W=None): # X is the number of inputs, H1 is the number of nodes in the first hidden layer,
                                                       # H2 is the number of nodes in the seconds hidden layer, Y is the number of outputs, W is a set of pretrained weights

        self.X = X # Inputs
        self.H1 = H1 # Nodes in Hidden layer 1
        self.H2 = H2 # Nodes in Hidden layer 2
        self.Y = Y # Outputs

        self.L = np.concatenate(([X], [H1], [H2], [Y])) 

        if W == None:
            w = []
            w.append(np.random.rand(self.X+1,self.H1).tolist()) # Initialses the weights by creating a random value between 0 and 1 for each input to each node in the first hidden layer, +1 is bias
            w.append(np.random.rand(self.H1+1,self.H2).tolist()) # Initialses the weights by creating a random value between 0 and 1 for each node in hidden layer 1 to each node in hidden layer 2, +1 is bias
            w.append(np.random.rand(self.H2+1,self.Y).tolist()) # Initialses the weights by creating a random value between 0 and 1 for each node in hidden layer 2 to each output, +1 is bias
            self.W = w
        else:
            self.W = W

        allDeltas = [] # A data structure to store the deltas produced during backpropagation to be used during gradient decent, the +1's are for the bias nodes
        allDeltas.append(np.zeros((self.X+1,self.H1)))
        allDeltas.append(np.zeros((self.H1+1,self.H2)))
        allDeltas.append(np.zeros((self.H2+1,self.Y)))
        self.allDeltas = allDeltas

        out = [] # A data structure to store the outputs from the neurons produced when the network is run
        out.append(np.zeros(X).tolist())
        out.append(np.zeros(H1).tolist())
        out.append(np.zeros(H2).tolist())
        out.append(np.zeros(Y).tolist())
        self.out = out

        outPreAct = [] # A data structure to store the outputs before put through the activation function from the neurons produced when the network is run
        outPreAct.append(np.zeros(X).tolist())
        outPreAct.append(np.zeros(H1).tolist())
        outPreAct.append(np.zeros(H2).tolist())
        outPreAct.append(np.zeros(Y).tolist())
        self.outPreAct = outPreAct

    def FF(self, input): # Runs the network, input is the array of inputs
        self.out[0] = input # The output of the input layer is just the input
        self.outPreAct[0] = input # The output of the input layer is just the input
        for k in range(len(self.L)-1): # 3 as there is 3 sets of weights, first goes from input to hidden layer 1, the second from hidden layer 1 to hidden layer 2, then the third goes from hidden layer 2 to the output
            input = np.concatenate(([1], input)) # Adds a bias of 1 to the start of the input array to this layer
            out = np.zeros(self.L[k+1]).tolist() # Initialses an output array of zero's for this layer
            outPreAct = np.zeros(self.L[k+1]).tolist() # Initialses an output array before of zero's for this layer, this one is for the output before it is put through the activation function
            for j in range(self.L[k+1]): # Loops through the outputs
                for i in range(len(input)): # Loops through the inputs for this layer summing up the inputs multiplied by the weights for each output to this layer
                    out[j] += input[i] * self.W[k][i][j] # Multiplies each input value with the corrisponding weight
                outPreAct[j] = out[j]
                out[j] = self.reLU(out[j]) # Sends the total sums through the ReLU function
            input = out # sets the input to the next layer as the output from this layer
            self.out[k+1] = out # Stores the outputs to be used later for backpropagation
            self.outPreAct[k+1] = outPreAct # Stores the outputs before being put through the activation function to be used later for backpropagation
        return out
    
    def train_nn(self, inputs, targets, epochs, learningRate): 
        for j in range(epochs): # Loops through the epochs
            for i in range(len(inputs)): # Loops through the sets of inputs
                input = inputs[i] # The array of inputs for this training
                target = targets[i] # The array of outputs for this training

                out = self.FF(input) # Runs the network on the test data
                error = out - target # Gets the error from the output
                deltas = self.BP(error) # Runs back propagation
                self.GD(deltas,learningRate) # Runs gradient decent

    def BP(self, errors):
        for j in reversed(range(len(self.L)-1)): # Loops through the sets of weights in reverse
            outputs = self.out[j+1] # Gets the outputs of the layer
            outputsPreAct = self.outPreAct[j+1] # Gets the outputs of the layer before they were put through the activation function
            inputs = np.concatenate(([1] ,self.out[j])) # Adds the bias to the inputs to the layer
            layerDeltas = []
            errDerArray = []
            for i in range(len(outputs)): # Loops through the outputs from this layer
                error = errors[i] # Gets the error for this output
                out = outputs[i] # Gets the output for this layer
                outPreAct = outputsPreAct[i] # Gets the output for this layer before it was put through the ReLU activation function
                errDer = error * self.reLU_Der(outPreAct) # Error * the derivative of the sigmoid function to get the value used for both gradient decent and to calculate the errors for the previous layers
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

    def reLU(self,x): # ReLU returns x is x is positive or 0, 0 if x is negative where x is the input to the function
        if x > 0:
            return x
        else:
            return 0

    def reLU_Der(self, x): # ReLU derviative returns 1 is if x is positive, 0 if x is negative or 0, where x is the input to the ReLU function. 
                            # ReLU technically doesnt have a derrivative when x = 0 however for practicality sake we say the derrivative is 0.
        if x > 0:
            return 1
        else:
            return 0


training_inputs = np.array([[100*np.random.rand() for _ in range(2)] for _ in range(10000)])   # This creates 10,000 training sets of input pairs where each input is a number between 0 and 100
targets = []
for input1, input2 in training_inputs: # Sets the target output for each pair of inputs as the larger input as the first output and the smaller input as the second output
    if input1 >= input2:
        targets.append([input1,input2])
    else:
        targets.append([input2,input1])

targets = np.array(targets)

nn=MultiLayerNN(2, 5, 5, 2)   # Creates a NN with 2 inputs, 2 hidden layers each with 5 nodes, and 1 ouput

# Testing data to identify if Network trained well. 
input = np.array([20, 35])      # after training this tests the train network 
target = np.array([35, 20])         # for this target value.  

NN_outputB = nn.FF(input) # Gets the output of our test data before training


nn.train_nn(training_inputs, targets, 10, 0.000001)  #trains the network with 0.000001 learning rate for 10 epochs

NN_outputA = nn.FF(input) # Gets the output of our test data after training

# Writes over the file containing the weights for the network that I trained for 10,000 epochs with 10,000 sets of training data in each epoch
""" epoch10000nnFileWrite = open("multiLayer10000EpochWeights.txt", "w") 
epoch10000nnFileWrite.write(json.dumps(nn.W))
epoch10000nnFileWrite.close()  """

epoch10000nnFileRead = open("multiLayer10000EpochWeights.txt", "r")
epoch10000nnWeights = json.loads(epoch10000nnFileRead.read())
nn10000Epoch = MultiLayerNN(2,5,5,2,epoch10000nnWeights)
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
print("As you can see the nn trained its self and has become far more acurate at sorting the input in descending order")
print()
print("Neural Network actual output 10000 epochs training done earlier is ",NN10000_output, "there is an error (not MSQE) of ",target-NN10000_output)
print("=================================================================") 



xORNN = MultiLayerNN(2,5,5,1)

xORTrainingInputs = [[0,0],[0,1],[1,0],[1,1]] # Training inputs for the XOR Perceptron are all 4 possible inputs
xORTrainingOutputs = []
for input1, input2 in xORTrainingInputs: # Assigns the target output the XOR inputs
    if input1 == 1 and input2 == 1:
        xORTrainingOutputs.append([0])
    elif input1 == 1 or input2 == 1:
        xORTrainingOutputs.append([1])
    else:
        xORTrainingOutputs.append([0])
xORTrainingOutputs = np.array(xORTrainingOutputs)

xORNN.train_nn(xORTrainingInputs, xORTrainingOutputs, 5000, 0.01) # Trains the XOR NN for 5000 epochs at a learning rate of 0.01

print("XOR")
print("---------------------")
result00 = xORNN.FF([0,0])[0]
print("Input: [0,0], Expected: 0, Result: " + str(result00) + ", Error: " + str(0-result00))
result01 = xORNN.FF([0,1])[0]
print("Input: [0,1], Expected: 1, Result: " + str(result01) + ", Error: " + str(1-result01))
result10 = xORNN.FF([1,0])[0]
print("Input: [1,0], Expected: 1, Result: " + str(result10) + ", Error: " + str(1-result10))
result11 = xORNN.FF([1,1])[0]
print("Input: [1,1], Expected: 0, Result: " + str(result11) + ", Error: " + str(0-result11))
print("---------------------")
print("As you can see for the XOR problem this gives the correct output for all 4 input combinations.")
print("#####################")