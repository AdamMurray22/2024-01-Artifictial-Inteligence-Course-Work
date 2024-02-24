import numpy as np

class Perceptron(object):

    def __init__(self, X=2): # X is the number of inputs
        w = [np.random.rand() for _ in range(X+1)] # Initialses the weights by creating a random value between 0 and 1 for each input, X+1 as the +1 is the bias weight
        self.W = w

    def run(self, input): # Runs the perceptron, input is the array of inputs
        input = np.concatenate(([1], input)) # Adds a bias of 1 to the start of array
        out = 0
        for i in range(len(input)): # Loops through the inputs summing up the inputs multiplied by the weights
            out += input[i] * self.W[i] # Multiplies each input value with the corrisponding weight, including multiplying the bias by 1
        out = self.threshold(out) # Sends the total sum through the threshold function to get a binary output
        return out
    
    def train_nn(self, inputs, targets, epochs, learningRate): # Trains the perceptron with the given array of sets of inputs, the target values for the inputs, the number of epochs and the learning rate
        for _ in range(epochs): # Loops for the number of epochs, runs the inputs through the perceptron epoch times
            for i in range(len(inputs)): # Loops through the input/target pairs and trains the perceptron
                input = inputs[i] # Array of inputs
                target = targets[i] # Target value for the array of inputs

                out = self.run(input) # Runs the perceptron
                input = np.concatenate(([1], input)) # Adds the bias of 1 to the start of the array
                error = target - out # Calculates the error, target - the output of the perceptron
                for j in range(len(self.W)): # Loops through the weights
                    self.W[j] = self.W[j] + (error * input[j] * learningRate) # Adjusts the weights, if the value from the perceptron was correct then no weights will change as error will be 0. 
                                                                               # When the error is not 0, the weights are adjusted by doing the error multiplied by the input to this weight,
                                                                               # multiplied by the learning rate.

    def threshold(self, out): # Returns a binary output, 1 if the value is above the threshold, otherwise 0
        if out > 0.5: # Threshold of 0.5
            return 1
        return 0
    
oRPerceptron = Perceptron(2) # Creates a perceptron with 2 inputs
xORPerceptron = Perceptron(2) # Creates a perceptron with 2 inputs
anyGreaterThan5Perceptron = Perceptron(3) # Creates a perceptron with 3 inputs

oRTrainingInputs = [[0,0],[0,1],[1,0],[1,1]] # Training inputs for the OR Perceptron are all 4 possible inputs
oRTrainingOutputs = []
for input1, input2 in oRTrainingInputs: # Assigns the target output the OR inputs
    if input1 == 1 or input2 == 1:
        oRTrainingOutputs.append(1)
    else:
        oRTrainingOutputs.append(0)

xORTrainingInputs = [[0,0],[0,1],[1,0],[1,1]] # Training inputs for the XOR Perceptron are all 4 possible inputs
xORTrainingOutputs = []
for input1, input2 in xORTrainingInputs: # Assigns the target output the XOR inputs
    if input1 == 1 and input2 == 1:
        xORTrainingOutputs.append(0)
    elif input1 == 1 or input2 == 1:
        xORTrainingOutputs.append(1)
    else:
        xORTrainingOutputs.append(0)
    
anyGreaterThan5TrainingInputs = np.array([[np.random.rand() for _ in range(3)] for _ in range(10000)]) # This creates 10,000 training sets of input pairs where each input is a number between 0 and 1
anyGreaterThan5TrainingOutputs = []
for input1, input2, input3 in anyGreaterThan5TrainingInputs: # Assigns the target output the any greater than 0.5 inputs
    if input1 > 0.5 or input2 > 0.5 or input3 > 0.5: # If any of the inputs are larger than 0.5, output 1, otherwise oupt 0
        anyGreaterThan5TrainingOutputs.append(1)
    else:
        anyGreaterThan5TrainingOutputs.append(0)

anyGreaterThan5Input1 = np.random.rand() # Generates test data for the any greater than 0.5 perceptron
anyGreaterThan5Input2 = np.random.rand()
anyGreaterThan5Input3 = np.random.rand()
if anyGreaterThan5Input1 > 0.5 or anyGreaterThan5Input2 > 0.5 or anyGreaterThan5Input3 > 0.5:
    anyGreaterThan5Output = 1
else:
    anyGreaterThan5Output = 0

oRPerceptron.train_nn(oRTrainingInputs, oRTrainingOutputs, 10000, 0.1) # Trains the OR Perceptron for 10000 epochs at a learning rate of 0.1
xORPerceptron.train_nn(xORTrainingInputs, xORTrainingOutputs, 10000, 0.1) # Trains the XOR Perceptron for 10000 epochs at a learning rate of 0.1
anyGreaterThan5Perceptron.train_nn(anyGreaterThan5TrainingInputs, anyGreaterThan5TrainingOutputs, 1, 0.1) # Trains the any value greater than 0.5 Perceptron for 1 epochs at a learning rate of 0.1

print("#####################")
print("OR Perceptron")
print("---------------------")
print("Input: [0,0], Expected: 0, Perceptron Result: " + str(oRPerceptron.run([0,0])))
print("Input: [0,1], Expected: 1, Perceptron Result: " + str(oRPerceptron.run([0,1])))
print("Input: [1,0], Expected: 1, Perceptron Result: " + str(oRPerceptron.run([1,0])))
print("Input: [1,1], Expected: 1, Perceptron Result: " + str(oRPerceptron.run([1,1])))
print("---------------------")
print("---------------------")
print("XOR Perceptron")
print("---------------------")
print("Input: [0,0], Expected: 0, Perceptron Result: " + str(xORPerceptron.run([0,0])))
print("Input: [0,1], Expected: 1, Perceptron Result: " + str(xORPerceptron.run([0,1])))
print("Input: [1,0], Expected: 1, Perceptron Result: " + str(xORPerceptron.run([1,0])))
print("Input: [1,1], Expected: 0, Perceptron Result: " + str(xORPerceptron.run([1,1])))
print("---------------------")
print("As you can see the OR Perceptron gives the correct output for all 4 input combinations, however the XOR Perceptron does not.")
print("This is due to OR being a linearly separable problem whilst XOR is not.")
print("---------------------")
print("---------------------")
print("However, if we interpret OR as A or B or C instead of the logical OR function the percptron can still solve the problem.")
print("The following is a percptron trained to output a 1 if any of its 3 inputs are greater than 0.5, otherwise it will output a 0.")
print("---------------------")
print("Input: [" + str(anyGreaterThan5Input1) + "," + str(anyGreaterThan5Input2) + "," + str(anyGreaterThan5Input3) + "], Expected: " + str(anyGreaterThan5Output) + ", Perceptron Result: " + str(anyGreaterThan5Perceptron.run([anyGreaterThan5Input1,anyGreaterThan5Input2,anyGreaterThan5Input3])))
print("#####################")