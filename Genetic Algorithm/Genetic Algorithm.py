import numpy as np
import time

def generatePop(popSize): # Generates a random population by generating a random number (either 1 or 0) for every allele in every individual
    pop = [] # Initialises the population as an empty array
    popValues = np.random.randint(2, size=(popSize,32)) # Generates an array of the size we want our population than contains arrays of random numbers (1 or 0),
                                                         # these sub arrays are our chromosomes for each individual
    for ind in range(popSize): # This converts our array of arrays of 1s and 0s to an array of strings of 1s and 0s, basically it converts all the chromosomes from arrays to strings
        pop.append(''.join(str(x) for x in popValues[ind])) # Converts the individual from an array of integers to a string and adds it to the population
    return pop

def generatePopAllZeros(popSize): # Generates a population where each string is 32 zero's for testing purposes
    ind = np.full(32, "0")
    indStr = ''.join(str(x) for x in ind)
    pop = np.full(popSize, indStr)
    return pop

def getIndividualFitness(ind): # Calculates an individuals fitness by summing up all the 1's in the individual. aka 000 would have a fitness of 0,
    count = 0                   # 101 would have a fitness of 2, 111 would have a fitnes of 3, thus the maximum fitness score is 32 for an individual of 32 1's and zero 0's
    for c in ind:
        if c == "1":
            count += 1
    return count

def producePopulationFitness(pop): # Calculates the fitness for all individuals in the population by looping through the population and applying the getIndividualFitness() function to each of them
    fit = []
    for i in range(len(pop)):
        fit.append([pop[i], getIndividualFitness(pop[i])])
    return fit

def rouletteFunction(pop): # Performs the roulette wheel selection method to select the parents for the next generation
    fitPop = producePopulationFitness(pop) # Calculates fitness for all individuals in the population
    sum = 0
    for _, fit in fitPop: # Sums the total fitness
        sum += fit
    currentProbabilityTotal = 0
    probabilityPop = []
    for ind, fit in fitPop: # Loops through the population and calculates an individuals probability of being selected by making there probablility proportional to the individuals fitness
        if sum == 0: # Base case just in case all of the population have a fitness of zero
            probability = 1
        else:
            probability = currentProbabilityTotal + fit / sum # The probability of an individual being selected is their fitness / the sum of all the fitness from the popualtion.
                                                                # To Understand why we add the currentProbabilityTotal lets use an example of moddeling flipping a coin. 0.5 chance of heads and 0.5 chance of tails.
                                                                # However if you generate a random number between 0 and 0.5, have you picked heads or tails, you dont know,
                                                                # so instead you say everything below and including 0.5 is heads, and above 0.5 up until an including 1 is tails,
                                                                # then you just generate a random number between 0 and 1 to get your coin flip
                                                                # Thats all the roulette wheel selection is just with many more outcomes and instead of 0.5 odds for everything,
                                                                # the odds for each thing are proportional to its fitness
        currentProbabilityTotal = probability # Updates the running total probability
        probabilityPop.append([ind, fit, probability])
    probabilitySum = currentProbabilityTotal # Total probability, should be 1 except where all individuals have a fitness of zero ignoring any errors introduced by the floating point calculations
    selectedPopParents = []
    randNums = np.random.uniform(0, probabilitySum, size=(len(pop), 2)) # Generates an array of pairs of random numbers within our range to select the parents,
                                                                         # an individual is selected to become a parent if the random number is less than its probability value
                                                                         # but greater than the previous individuals probability value
    for ind in range(len(pop)): # Loops until 2 parents have been selected for each future child
        parents = []
        for parent in range (2): # Loops twice to select 2 parents
            randNum = randNums[ind][parent] # Gets the random number for this parent from the array of pairs generated befor
            if randNum <= probabilityPop[0][2]: # This is just a base case if it selects the first individual from the previous generation to become a parent
                parents.append(probabilityPop[0][0])
            for i in range(1, len(probabilityPop)): # Loops through the rest of the population to find the parent
                if randNum > probabilityPop[i - 1][2] and randNum <= probabilityPop[i][2]: # Checks if the random number is greater than the previous individuals probability value 
                                                                                            # but less than the current individuals probability value
                    parents.append(probabilityPop[i][0])
                    break
        selectedPopParents.append(parents)
    return selectedPopParents

def crossover(selectedPopParents): # Performs single point crossover for each pair of parents to create a new population
    childPop = []
    randNumbers = np.random.randint(0, 33, size=len(selectedPopParents)) # Generates a random number, 0-32 inclusive, and then uses this number as the crossover point,
                                                                          # if its 0 the child will entirely be from parent 2,
    for childNum in range(len(selectedPopParents)): # Loops through the pairs of parents to seperatly perform crossover on each pair
        parent1 = selectedPopParents[childNum][0] # Gets the first parent for this child from the list
        parent2 = selectedPopParents[childNum][1] # Gets the seecond parent for this child from the list
        crossoverPoint = randNumbers[childNum] # Gets the random number for this childs crossover point from the array generated before
        child = parent1[:crossoverPoint] + parent2[crossoverPoint:] # The child is produced by taking all alleles in the index's before the crossover point from parent 1 and all the alleles
                                                                     # in the index's after and including the crossover point from parent 2
        childPop.append(child)
    return childPop
    
def mutation(pop, mutationRate): # Performs mutation upon the child population
    newPop = []
    randNumbers = np.random.rand(len(pop), 32) # Generates an array of arrays of random numbers between 0 and 1. The inner arrays are of length 32 as each random number corrisponds to an allele.
    for indNum in range(len(pop)): # Loops through the population checking each allele in eeach individual to see if it has been mutated
        ind = pop[indNum]
        newInd = []
        for i in range(len(ind)): # Loops through each allele in the curreent individual
            randNum = randNumbers[indNum][i] # Gets the random number corrisponding to the allele in this individual
            if randNum < mutationRate: # If the random number is less than the mutation rate then a mutaion occurs, thus the higher mutation rate the more mutations will happen
                if ind[i] == "0": # If the allele contained a 0 it becomes a 1
                    newInd.append(1)
                else:
                    newInd.append(0) # If the allele contained a 1 it becomes a 0
            else:
                newInd.append(ind[i])
        newPop.append(''.join(str(x) for x in newInd))
    return newPop

popSize = 100
maxGenerations = 1000 # Minimum 2 as it will always generate population and run the genetic algorithm once.
mutationRate = 0.0015 # 0.15 percent

pop = generatePop(popSize) # Generates a random population
#pop = generatePopAllZeros(popSize) # Creates a population where each string is 32 zero's for testing purposes

print('###########################') 
print('Population generated of ' + str(popSize))

loop = True
genCount = 0
bestFit = []

popFit = [[getIndividualFitness(pop[i]), pop[i]] for i in range(len(pop))] # Gets the fitness for our initial population
bestFitPop = max([popFit[i] for i in range(len(popFit))]) # Finds the max fitness
bestFit.append(bestFitPop) # Adds the max to an array tracking the best fitness from each generation
genCount +=1

print('________________________')
print('Initial population and Corresponding Fitness')
print('________________________')
for fit, ind in popFit:
    print(ind + ", " + str(fit))
print('________________________')
print("Best initial individual is:")
print(bestFitPop[1] + ", " + str(bestFitPop[0]))

while loop:
    # This first part is the genetic algorithm, first roulette wheel selection (which calculats fitness), then single point crossover, then mutation
    selectedPopParents = rouletteFunction(pop) # Performs a  roulette selection and returns an array of length of the population with each entry containing pairs of parents
    childPop = crossover(selectedPopParents) # Performs single-point crossover on the parents array to generate children from these parents
    newPop = childPop # Here so when the line below is commented out to demonstrate working selection/crossover the code will still work
    newPop = mutation(childPop, mutationRate) # Mutates the child population to produce our new population
    pop = newPop

    # This part just finds the best of the generation, stores it and outputs it 
    popFit = [[getIndividualFitness(pop[i]), pop[i]] for i in range(len(pop))] # Gets the fitness for the new children
    bestFitPop = max([popFit[i] for i in range(len(popFit))]) # Finds the max fitness
    bestFit.append(bestFitPop) # Adds the max to our array tracking the best fitness from each generation
    genCount += 1
    print('________________________')
    print("Best Fit in Gen " + str(genCount) + ": " + bestFitPop[1] + ", " + str(bestFitPop[0]))

    if (genCount >= maxGenerations or bestFitPop[0] == 32): # Terminates the algorithm if either the max generations has been reached or a child consisting of 32 1's is found 
        loop = False

print('________________________')
print('________________________')
print("Best fit from Generation 1:")
print(bestFit[0][1] + ", " + str(bestFit[0][0])) # Outputs the best individual from the initial generation
print('________________________')
print("Best fit from the final(Gen " + str(genCount) + ") Generation:")
print(bestFit[len(bestFit) - 1][1] + ", " + str(bestFit[len(bestFit) - 1][0])) # Outputs the best individual from the final generation
print('________________________')
print("This produced a fitness increase of: " + str(bestFit[len(bestFit) - 1][0] - bestFit[0][0]))
print('###########################') 

# To demonstrate working mutation change the generated population to all zero's but uncommenting line 106 
# To demonstrate working selction/crossover comment out mutation on line 134